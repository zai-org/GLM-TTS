# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
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
import os
import glob
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Import shared constants and utilities from glmtts_inference
from glmtts_inference import (
    DEVICE,
    load_frontends,
    yaml_util,
    tts_model_util,
    SpeechTokenizer
)

def load_flow_only(frontend_dir, flow_ckpt, flow_config, sample_rate=24000):
    """
    Loads only the Flow model and necessary frontends, skipping the LLM.
    """
    print(f"[INFO] Loading Speech Tokenizer and Frontends (SR={sample_rate}) from {frontend_dir}...")
    
    # Load Speech Tokenizer
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(os.path.join('ckpt', 'speech_tokenizer'))
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)
    
    # Load Frontends with specific sample_rate
    frontend, _ = load_frontends(
        speech_tokenizer, 
        use_phoneme=False, 
        sample_rate=sample_rate, 
        frontend_dir=frontend_dir
    )

    print(f"[INFO] Loading Flow Model from {flow_ckpt}...")
    flow = yaml_util.load_flow_model(flow_ckpt, flow_config, DEVICE)
    
    # Wrap in Token2Wav for inference (Ensure Token2Wav gets the SR)
    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)
    
    return frontend, token2wav

def process_prompt(frontend, prompt_wav_path, sample_rate=24000):
    """
    Extracts speaker embedding, speech tokens, and speech features from the prompt audio.
    """
    if not os.path.exists(prompt_wav_path):
        raise FileNotFoundError(f"Prompt audio not found: {prompt_wav_path}")

    print(f"[INFO] Extracting features from prompt: {prompt_wav_path}")
    
    # Extract features using the frontend
    prompt_token = frontend._extract_speech_token([prompt_wav_path]).to(DEVICE)
    embedding = frontend._extract_spk_embedding(prompt_wav_path).to(DEVICE)
    
    # Pass sample_rate to feature extraction
    prompt_feat = frontend._extract_speech_feat(prompt_wav_path, sample_rate=sample_rate).to(DEVICE)
    
    return prompt_token, prompt_feat, embedding

def extract_tokens_from_audio(frontend, audio_path):
    """
    Extract speech tokens from an audio file on the fly.
    Returns a list of integers.
    """
    # frontend._extract_speech_token returns a tensor of shape (Batch, Seq)
    # Since we pass a single path, Batch is 1.
    token_tensor = frontend._extract_speech_token([audio_path])
    
    # Convert to standard list of integers
    token_list = token_tensor.squeeze().tolist()
    
    # Handle single token case just in case (though unlikely for speech)
    if isinstance(token_list, int):
        token_list = [token_list]
        
    return token_list

def reconstruct_audio(token2wav, tokens, prompt_data):
    """
    Reconstructs audio from tokens using the Flow model and prompt data.
    """
    prompt_token, prompt_feat, embedding = prompt_data
    
    with torch.no_grad():
        wav, _ = token2wav.token2wav_with_cache(
            tokens,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding
        )
    
    return wav.detach().cpu()

def main():
    parser = argparse.ArgumentParser(description="GLMTTS Flow Reconstruction Script")
    
    # Path Arguments
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory containing .npy token files OR audio files.")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Directory to save reconstructed .wav files.")
    parser.add_argument("--prompt_wav", "-p", type=str, required=True, help="Path to the prompt audio file (for speaker timbre).")
    
    # Model Arguments
    parser.add_argument("--flow_ckpt", type=str, default="ckpt/flow/flow.pt", help="Path to Flow model checkpoint.")
    parser.add_argument("--flow_config", type=str, default="ckpt/flow/config.yaml", help="Path to Flow model config.")
    parser.add_argument("--frontend_dir", type=str, default="frontend", help="Path to frontend resources.")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Output sample rate (24000 or 32000).")
    
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "_flow_recon")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Models
    frontend, token2wav = load_flow_only(
        args.frontend_dir, 
        args.flow_ckpt, 
        args.flow_config, 
        sample_rate=args.sample_rate
    )

    # 2. Process Prompt
    try:
        prompt_data = process_prompt(frontend, args.prompt_wav, sample_rate=args.sample_rate)
    except Exception as e:
        print(f"[ERROR] Failed to process prompt: {e}")
        return

    # 3. Determine Input Mode (NPY or Audio)
    # Priority: .npy -> Audio
    input_files = glob.glob(os.path.join(args.input_dir, '*.npy'))
    mode = 'npy'
    
    if not input_files:
        # If no npy found, look for audio
        audio_exts = ['*.wav']
        input_files = []
        for ext in audio_exts:
            input_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        mode = 'audio'
        
        # Exclude the prompt file itself if it's in the same directory to avoid redundancy
        prompt_abs = os.path.abspath(args.prompt_wav)
        input_files = [f for f in input_files if os.path.abspath(f) != prompt_abs]

    if not input_files:
        print(f"[WARN] No .npy or supported audio files found in {args.input_dir}")
        return

    print(f"[INFO] Mode: {mode.upper()}. Found {len(input_files)} files to process.")

    # 4. Processing Loop
    for file_path in tqdm(input_files, desc="Reconstructing"):
        try:
            base_name = os.path.basename(file_path)
            # Create output name, ensuring we don't overwrite input if dir is same (though output_dir handles this usually)
            name_stem = os.path.splitext(base_name)[0]
            output_name = name_stem + '_recon.wav' if mode == 'audio' else name_stem + '.wav'
            output_path = os.path.join(args.output_dir, output_name)

            token_list = []

            if mode == 'npy':
                # Load tokens from file
                token_list = np.load(file_path).tolist()
                token_list = [int(t) for t in token_list if t >= 0]
            else:
                # Extract tokens on the fly (Audio Mode)
                token_list = extract_tokens_from_audio(frontend, file_path)

            if not token_list:
                print(f"[SKIP] Empty token list extracted from {base_name}")
                continue

            # Reconstruct
            waveform = reconstruct_audio(token2wav, token_list, prompt_data)

            # Post-process waveform shape
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Save
            torchaudio.save(output_path, waveform, args.sample_rate)

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"[INFO] Reconstruction complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()