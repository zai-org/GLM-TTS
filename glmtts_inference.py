# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
# Authors: Jiayan Cui, Zhihan Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import logging
import os
import torch
import torchaudio
import tqdm

from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import file_utils, seed_util
from utils import tts_model_util, yaml_util
from transformers import AutoTokenizer, LlamaForCausalLM
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from functools import partial
# --- Global Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750
TOKEN_RATE = 25
EOS_TOKEN_ID_AFTER_MINUS_BOS = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_special_token_ids(tokenize_fn):
    """
    Get special token IDs based on the tokenizer name.
    """
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    special_token_ids = {}

    # Validation
    endoftext_id = tokenize_fn("<|endoftext|>")[0]
    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        # Check 1: Special token length must be 1
        if len(__ids) != 1:
            raise AssertionError(
                f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}"
            )
        # Check 2: Special token ID must be >= endoftext_id
        if __ids[0] < endoftext_id:
            raise AssertionError(
                f"Token '{k}' ({v}) ID {__ids[0]} is smaller than endoftext ID {endoftext_id}"
            )

        special_token_ids[k] = __ids[0]

    return special_token_ids

def _assert_shape_and_get_len(token):
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"):
    if sample_rate == 32000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        print("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)
        print("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join('ckpt', 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )

    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend


def local_llm_forward(
    llm,
    prompt_text_token,
    tts_text_token,
    prompt_speech_token,
    beam_size=1,
    sampling=25,
    sample_method="ras",
):
    """
    Single LLM forward pass.
    """
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,  # No specific speaker embedding needed for generic pretrain inference here
    )
    return tts_speech_token[0].tolist()


def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """
    Single Flow forward pass.
    """
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel


# --- Helper Function: Get Prompt from Cache ---
def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """
    Constructs prompt tokens from the cache.
    Prunes the cache if the sequence length exceeds MAX_LLM_SEQ_INP_LEN.
    """
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    # Estimate required length ratio
    # Avoid division by zero
    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)

    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # Prune cache if too long.
    # Logic: Keep the first item (original prompt), remove from the second item onwards.
    while (
        __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN
    ):
        if len(cache_speech_token) <= 1:
            break  # Always keep at least the original prompt
        # logging.debug(f'[get_cached_prompt] Cache pop. Text count before: {len(cache_text)}')
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # Construct Text Prompt
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # Construct Speech Prompt
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)

    return prompt_text_token, llm_speech_token


# --- Main Generation Logic ---


def generate_long(
    frontend: TTSFrontEnd,
    text_frontend: TextFrontEnd,
    llm,
    flow,
    text_info,
    cache,
    device,
    embedding,
    seed=0,
    sample_method="ras",
    flow_prompt_token=None,
    speech_feat=None,
    local_llm_forward=local_llm_forward,
    local_flow_forward=local_flow_forward,
    use_phoneme=False,
):
    outputs = []
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    short_text_list = text_frontend.split_by_len(syn_text)

    for _, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed)
        tts_text_tn = text_frontend.text_normalize(
            tts_text
        )  # Normalize again after splitting
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        # Access cache references
        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]

        # Determine Prompts
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(
                cache, tts_text_token, device
            )
        else:
            # Initial prompt case
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor(
                [cache_speech_token[0]], dtype=torch.int32
            ).to(device)
            logging.debug("[generate_long] Using initial prompt (empty cache history)")

        # LLM Inference
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method
        )

        output_token_list.extend(token_list_res)

        # Flow Inference
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding
        )

        # Update Cache
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        outputs.append(output)
        if full_mel is not None:
            full_mels.append(full_mel)

    tts_speech = torch.concat(outputs, dim=1)
    tts_mel = torch.concat(full_mels, dim=-1) if full_mels else None

    return tts_speech, tts_mel, output_token_list, text_tn_dict


def jsonl_generate(
    data_name, folder_path, sample_rate=24000, seed=0, use_cache=True, use_phoneme=False
):
    # Dataset path resolution
    jsonl_path = os.path.join("examples", data_name + ".jsonl")

    logging.info(f"Using jsonl: {jsonl_path}")
    item_list = file_utils.get_jsonl(jsonl_path)

    output_json_path = os.path.join(folder_path, "text_compare.jsonl")

    with open(output_json_path, "w") as f_out:
        for item in tqdm.tqdm(item_list):
            try:
                uttid = item["uttid"]
                wav_save_path = os.path.join(folder_path, f"{uttid}.wav")

                # Text Normalization
                prompt_text = text_frontend.text_normalize(item["prompt_text"])
                synth_text = text_frontend.text_normalize(item["syn_text"])

                prompt_text_token = frontend._extract_text_token(prompt_text+" ")
                prompt_speech_token = frontend._extract_speech_token(
                    [item["prompt_speech"]]
                )
                speech_feat = frontend._extract_speech_feat(item["prompt_speech"], sample_rate=sample_rate)
                embedding = frontend._extract_spk_embedding(item["prompt_speech"])
                cache_speech_token = [prompt_speech_token.squeeze().tolist()]
                flow_prompt_token = torch.tensor(
                    cache_speech_token, dtype=torch.int32
                ).to(DEVICE)

                # Initialize Cache
                cache = {
                    "cache_text": [prompt_text],
                    "cache_text_token": [prompt_text_token],
                    "cache_speech_token": cache_speech_token,
                    "use_cache": use_cache,
                }
                syn_text = item["syn_text"]
                logging.info(f"Processing: {uttid}, Syn_text: {syn_text}")

                # Run Generation
                tts_speech, _, _, text_tn_dict = generate_long(
                    frontend=frontend,
                    text_frontend=text_frontend,
                    llm=llm,
                    flow=flow,
                    text_info=[uttid, synth_text],
                    cache=cache,
                    embedding=embedding,
                    seed=seed,
                    flow_prompt_token=flow_prompt_token,
                    speech_feat=speech_feat,
                    device=DEVICE,
                    use_phoneme=use_phoneme,
                )
                f_out.write(
                    json.dumps(text_tn_dict, ensure_ascii=False, indent=2) + "\n"
                )
                f_out.flush()
                # Save Wave and Tokens
                os.makedirs(os.path.dirname(wav_save_path), exist_ok=True)
                torchaudio.save(wav_save_path, tts_speech, sample_rate)

                # Optinal: save prompt features as data input for RL
                # feat_root = os.path.join('grpo', 'data')

                # np.save(os.path.join(feat_root, 'prompt_speech_token', item['uttid']), prompt_speech_token.cpu().squeeze().numpy())
                # np.save(os.path.join(feat_root, 'prompt_speech_feat', item['uttid']), speech_feat.cpu().squeeze().numpy())
                # np.save(os.path.join(feat_root, 'embedding', item['uttid']), embedding.cpu().squeeze().numpy())

            except Exception as e:
                logging.error(f"Error processing {item.get('uttid', 'unknown')}: {e}")
                import traceback

                traceback.print_exc()
                # Optional: raise e # Uncomment to stop on first error


def load_models(use_phoneme=False, sample_rate=24000):
    # Load Speech Tokenizer
    speech_tokenizer_path = os.path.join("ckpt", "speech_tokenizer")
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(
        speech_tokenizer_path
    )
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # Load Frontends
    frontend, text_frontend = load_frontends(speech_tokenizer, sample_rate=sample_rate, use_phoneme=use_phoneme)

    llama_path = os.path.join("ckpt", "llm")

    llm = GLMTTS(
        llama_cfg_path=os.path.join(llama_path, "config.json"), mode="PRETRAIN"
    )
    llm.llama = LlamaForCausalLM.from_pretrained(
        llama_path, dtype=torch.float32
    ).to(DEVICE)

    llm.llama_embedding = llm.llama.model.embed_tokens

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    flow_ckpt = os.path.join("ckpt", "flow", "flow.pt")
    flow_config = os.path.join("ckpt", "flow", "config.yaml")
    flow = yaml_util.load_flow_model(
        flow_ckpt, flow_config, DEVICE
    )

    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)

    return frontend, text_frontend, speech_tokenizer, llm, token2wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLM-TTS Inference Script (Pretrain Mode Only)"
    )
    parser.add_argument("--data", default="example_zh", type=str)
    parser.add_argument("--exp_name", default="_test", type=str)
    parser.add_argument("--use_cache", action="store_true", default=True)
    parser.add_argument("--use_phoneme", action="store_true", default=False)
    parser.add_argument("--sample_rate", type=int, default=24000)

    args = parser.parse_args()

    # Load Models
    frontend, text_frontend, speech_tokenizer, llm, flow = load_models(
        use_phoneme=args.use_phoneme,
        sample_rate=args.sample_rate
    )

    # Create Output Directory
    folder_path = os.path.join(
        CURRENT_DIR, "outputs", f"pretrain{args.exp_name}", args.data
    )
    os.makedirs(folder_path, exist_ok=True)
    logging.info(f"Output folder: {folder_path}")

    # Run Inference
    jsonl_generate(
        args.data, folder_path, sample_rate=args.sample_rate, use_cache=args.use_cache, use_phoneme=args.use_phoneme
    )
