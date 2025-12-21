# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
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

import os
import re
import json
import random
from typing import Callable, List, Tuple, Union, Optional

import librosa
import inflect
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

# Local imports
from utils.glm_g2p import G2P_zh, process_one, is_chinese
from utils.file_utils import load_wav
from cosyvoice.utils.frontend_utils import (
    contains_chinese, remove_bracket, replace_asterisk_with_multiply,
    spell_out_number, tn_scientific_notation, split_hard,
    split_into_min_sentence, multi_line_process, PUNCTUATION_CHARS,
    emoji_norm, markdown_norm, normalize_punctuation, special_replace,
    ensure_proper_ending
)

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("Warning: failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False


class SpeechTokenizer:
    """
    Tokenizer for extracting discrete speech tokens from audio.
    """
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

    def extract_speech_token(self, utts: List[Union[str, Tuple[torch.Tensor, int]]]) -> List[List[int]]:
        assert isinstance(utts, list)

        _resample_buffer = self._resample_buffer
        model, feature_extractor = self.model, self.feature_extractor

        with torch.no_grad():
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                else:
                    audio, sample_rate = torchaudio.load(utt)

                audio = audio.cuda()

                # Resample to 16k if needed
                if sample_rate != 16000:
                    if sample_rate not in _resample_buffer:
                        _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate,
                            new_freq=16000
                        ).to('cuda')
                    audio = _resample_buffer[sample_rate](audio)

                audio = audio[0]  # Take first channel
                audio = audio.cpu().numpy()

                # Segment audio into 30s chunks to avoid OOM
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30

            pooling_kernel_size = model.config.pooling_kernel_size or 1
            stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 128

            for start in range(0, len(audios), batch_size):
                features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                             return_attention_mask=True, return_tensors="pt", device='cuda',
                                             padding="longest", pad_to_multiple_of=stride)
                features = features.to(device="cuda")
                outputs = model(**features)
                speech_tokens = outputs.quantized_token_ids

                attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
                attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
                assert attention_mask.shape == speech_tokens.shape

                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)
            return all_speech_tokens


class TextFrontEnd:
    """
    Text Frontend for handling Text Normalization (TN) and Grapheme-to-Phoneme (G2P).
    Supports mixed Chinese and English input.
    """
    def __init__(self, use_phoneme: bool = False):
        # Define constants
        self.PUNCTUATION_CHARS = PUNCTUATION_CHARS
        self.use_ttsfrd = use_ttsfrd

        if use_phoneme:
            self.text_tokenizer = G2P_zh()

        # Initialize TTS Frontend Engine
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            # Ensure the resource path exists
            res_path = os.path.normpath(os.path.join(ROOT_DIR, '../../pretrained_models/CosyVoice-ttsfrd/resource'))
            assert self.frd.initialize(res_path) is True, 'Failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyin')
            self.frd.enable_pinyin_mix(True)
            self.frd.set_breakmodel_index(1)
        else:
            self.zh_tn_model = ZhNormalizer(
                remove_erhua=False,
                full_to_half=True,
                remove_interjections=False,
                overwrite_cache=True
            )
            self.en_tn_model = EnNormalizer()

        self.use_phoneme = use_phoneme
        if self.use_phoneme:
            self.able_list = []
            script_path = os.path.abspath(__file__)
            # Navigate to configs directory
            use_phoneme_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_path))), "configs")
            able_path = os.path.join(use_phoneme_dir, "G2P_able_1word.json")

            with open(able_path, 'r', encoding='utf-8') as f:
                self.able_list = json.load(f)

            replace_dict_path = os.path.join(use_phoneme_dir, "G2P_replace_dict.jsonl")
            self.replace_dict = {}
            with open(replace_dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self.replace_dict.update(d)

        self.inflect_parser = inflect.engine()

    def text_normalize(self, text: str) -> Optional[str]:
        """
        Text Normalization Flow:
        1. Pre-processing: Handle emojis, markdown, multi-lines.
        2. Language-specific processing: Chinese vs English.
        3. Punctuation normalization.
        4. Ending punctuation enforcement.
        """
        if text is None:
            return None

        # 1. Pre-processing
        text = self._preprocess_text(text)

        # 2. Language-specific processing
        if contains_chinese(text):
            text = self._normalize_chinese_text(text).lower()
        else:
            text = self._normalize_english_text(text)

        # 3. Punctuation normalization
        text = normalize_punctuation(text, self.PUNCTUATION_CHARS)

        # 4. Ensure proper ending
        text = ensure_proper_ending(text)

        return text

    def _preprocess_text(self, text: str) -> str:
        """Text Pre-processing: Handle special formats and symbols."""
        # Handle Markdown
        text = markdown_norm(text)
        # Handle multi-line text
        text = multi_line_process(text)
        # Handle emojis
        text = emoji_norm(text)
        # If hyphen is between English characters, replace with space
        text = re.sub(r'(?<=[a-zA-Z])-(?=[a-zA-Z])', ' ', text)

        return text

    def _normalize_chinese_text(self, text: str) -> str:
        """Normalize Chinese text."""
        # Pre-replace special characters and formats
        text = self.pre_replace(text)

        # TTS Frontend processing
        if self.use_ttsfrd:
            text = self.frd.get_frd_extra_info(text, 'input')
        else:
            text = self.zh_tn_model.normalize(text)

        # Special character handling (edge cases)
        text = special_replace(text)

        text = self.post_replace(text)
        return text

    def pre_replace(self, sentence: str) -> str:
        """Replacements applied BEFORE normalizer."""
        # Scientific notation
        sentence = tn_scientific_notation(sentence)
        # Remove hyphen if both sides are not numbers.
        sentence = re.sub(r'(?<=\d)\s*-\s*(?=\d)', '减', sentence)
        sentence = sentence.replace('-', '')
        # Replace '咯' with '喽' when followed by punctuation
        sentence = re.sub(r'咯([' + re.escape(PUNCTUATION_CHARS) + r'])', r'喽\1', sentence)
        # Character variant replacement
        # Custom replacements (e.g. ancient poetry)
        custom_replace_path = './configs/custom_replace.jsonl'
        if os.path.exists(custom_replace_path):
            with open(custom_replace_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)
                    sentence = sentence.replace(line['origin'], line['new'])
        return sentence

    def post_replace(self, sentence: str) -> str:
        """Replacements applied AFTER normalizer."""
        # Bracket removal
        sentence = remove_bracket(sentence)
        # Punctuation normalization
        sentence = sentence.replace(" - ", "，")
        sentence = sentence.replace("——", "，")
        sentence = re.sub(r'[,:：;；、]+', '，', sentence)
        sentence = re.sub(r'[.…]+', '。', sentence)
        sentence = re.sub(r'[_·]+', '', sentence)
        sentence = re.sub(r"""['"‘’“”|]+""", '', sentence)

        # Special Symbol Mapping
        sentence = sentence.replace('†', '，')
        sentence = sentence.replace('²', '平方')
        sentence = sentence.replace('³', '立方')
        sentence = sentence.replace('/', '每')
        sentence = sentence.replace('~', '到')
        sentence = sentence.replace('～', '到')

        # Number circling mapping
        replacements = {
            '①': '一', '②': '二', '③': '三', '④': '四', '⑤': '五',
            '⑥': '六', '⑦': '七', '⑧': '八', '⑨': '九', '⑩': '十'
        }
        for k, v in replacements.items():
            sentence = sentence.replace(k, v)

        # Greek alphabet mapping
        greek_map = {
            'α': '阿尔法', 'β': '贝塔', 'γ': '伽玛', 'Γ': '伽玛',
            'δ': '德尔塔', 'Δ': '德尔塔', '△': '德尔塔', 'ε': '艾普西龙',
            'ζ': '捷塔', 'η': '依塔', 'θ': '西塔', 'Θ': '西塔',
            'ι': '艾欧塔', 'κ': '喀帕', 'λ': '拉姆达', 'Λ': '拉姆达',
            'μ': '缪', 'ν': '拗', 'ξ': '克西', 'Ξ': '克西',
            'ο': '欧米克伦', 'π': '派', 'Π': '派', 'ρ': '肉',
            'ς': '西格玛', 'Σ': '西格玛', 'σ': '西格玛', 'τ': '套',
            'υ': '宇普西龙', 'φ': '服艾', 'Φ': '服艾', 'χ': '器',
            'ψ': '普赛', 'Ψ': '普赛', 'ω': '欧米伽', 'Ω': '欧米伽',
            '□': '方框'
        }
        for k, v in greek_map.items():
            sentence = sentence.replace(k, v)

        # Math symbol mapping
        math_map = {
            '>': '大于', '<': '小于', '∈': '属于', '∉': '不属于',
            '∪': '并', '∩': '交', '⊥': '垂直', '∥': '平行',
            '≠': '不等于', '∵': '因为', '∴': '所以', '∅': '空集',
            '⊂': '真包含于', '⊃': '包含', '⊆': '包含于', '⊇': '真包含',
            '⊄': '不属于', '⊅': '非超集', '⊈': '不属于', '⊉': '非超集'
        }
        for k, v in math_map.items():
            sentence = sentence.replace(k, v)

        return sentence

    def _normalize_english_text(self, text: str) -> str:
        text = text.replace('\'', '’')
        """Normalize English text."""
        # TTS Frontend processing
        if self.use_ttsfrd:
            text = self.frd.get_frd_extra_info(text, 'input')
        else:
            text = self.en_tn_model.normalize(text)

        text = remove_bracket(text, 'en')
        text = replace_asterisk_with_multiply(text, 'en')
        text = text.replace('—', ' ')
        text = text.replace('’', '\'')
        # Spell out numbers
        text = spell_out_number(text, self.inflect_parser)

        # Expand contractions
        # text = contractions.fix(text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Filter punctuation
        keep_punctuation = r'\.,!\?\'\:;'
        pattern = rf'[^\w\s{keep_punctuation}]'
        text = re.sub(pattern, '', text)
        text = text.lower()

        # Normalize repeated punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\,+', ',', text)
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        text = re.sub(r'\'+', '\'', text)
        text = re.sub(r':+', ':', text)
        text = re.sub(r';+', ';', text)

        # Ensure correct punctuation spacing
        text = re.sub(r'\s*([.,?!\':;])\s*', r'\1 ', text)
        text = text.strip()
        return text

    def split_by_len(self, text: str, min_text_len: int = 30, max_text_len: int = 60) -> List[str]:
        """Split text by length constraints."""
        min_sentences, _ = split_into_min_sentence(text, min_text_len)
        sentence_x_units = split_hard(min_sentences, max_text_len)
        res = [''.join(units) for units in sentence_x_units]
        return res

    def _split_mixed_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split text into chunks of (content, is_chinese_flag).
        Ensures that consecutive Chinese characters are kept together for G2P context.
        """
        if not text:
            return []

        result = []
        current_chunk = ""
        # Initialize state
        is_current_chinese = is_chinese(text[0])

        for char in text:
            char_is_zh = is_chinese(char)
            if char_is_zh == is_current_chinese:
                current_chunk += char
            else:
                result.append((current_chunk, is_current_chinese))
                current_chunk = char
                is_current_chinese = char_is_zh

        if current_chunk:
            result.append((current_chunk, is_current_chinese))

        return result

    def _tokenize_by_replace_dict(self, text: str) -> List[Tuple[str, bool]]:
        """
        Apply custom replacement dictionary (replace_dict).
        Returns: List of (text_fragment, is_replaced_flag)
        """
        if not self.replace_dict:
            return [(text, False)]

        sorted_keys = sorted(self.replace_dict, key=len, reverse=True)
        i, n = 0, len(text)
        fragments = []

        while i < n:
            matched = False
            for key in sorted_keys:
                if text.startswith(key, i):
                    fragments.append((self.replace_dict[key], True))
                    i += len(key)
                    matched = True
                    break
            if not matched:
                # Accumulate unmatched characters
                if fragments and not fragments[-1][1]:
                    fragments[-1] = (fragments[-1][0] + text[i], False)
                else:
                    fragments.append((text[i], False))
                i += 1
        return fragments

    def _format_phonemes(self, phoneme_parts: list) -> str:
        """
        Format phoneme list (e.g., ['sh', '|', 'ang']) into wrapped string "<|SH|><|ANG|>".
        """
        # 1. To Upper Case
        # 2. Remove '-'
        # 3. Wrap with <||>
        res = []
        for p in phoneme_parts:
            p = p.strip()
            if not p:
                continue
            if p == '|':
                continue  # Skip raw separators

            p_upper = p.upper().replace("-", " ")
            if p_upper in self.PUNCTUATION_CHARS:
                res.append(p_upper)
            else:
                res.append(f"<|{p_upper}|>")
        return "".join(res)

    def _align_and_replace(self, text_chunk: str, phoneme_list: list) -> str:
        """
        Alignment algorithm:
        Input: text_chunk="你好吗？", phoneme_list=['n', '|', 'i', '-', 'h', '|', 'ao', '-', 'm', '|', 'a', '？']
        Output: Decide whether to keep the character or replace with phonemes based on able_list.
        """
        result = []
        ph_idx = 0
        total_ph = len(phoneme_list)

        for char in text_chunk:
            # If punctuation, G2P usually matches it, so consume one token.
            if char in self.PUNCTUATION_CHARS:
                result.append(char)
                # Attempt to consume the corresponding punctuation in phoneme_list
                # This is a fault-tolerance check.
                if ph_idx < total_ph and phoneme_list[ph_idx] in self.PUNCTUATION_CHARS:
                    ph_idx += 1
                continue

            # If Chinese character, collect phonemes until '-' or punctuation or end.
            current_char_phones = []

            while ph_idx < total_ph:
                token = phoneme_list[ph_idx]

                # 1. Syllable separator '-', marks end of current character
                if token == '-':
                    ph_idx += 1
                    break

                # 2. Punctuation
                # glm_g2p logic: if followed by punctuation, the preceding '-' might be omitted.
                # If we meet punctuation, stop current char collection (leave punctuation for next loop).
                if token in self.PUNCTUATION_CHARS:
                    break

                # 3. Normal phoneme part
                current_char_phones.append(token)
                ph_idx += 1

            # Decision: Replace or Keep
            # If character is NOT in whitelist -> Replace with phoneme tokens
            if char not in self.able_list:
                # Only replace if phonemes were actually collected
                if current_char_phones:
                    result.append(self._format_phonemes(current_char_phones))
                else:
                    result.append(char)  # Fallback: keep original
            else:
                # In whitelist -> Keep original character
                result.append(char)

        return "".join(result)

    def g2p_infer(self, text: str) -> str:
        """
        G2P Inference Pipeline:
        1. Apply dictionary replacements.
        2. Split remaining text into [Chinese_Block, Non_Chinese_Block, ...].
        3. For Chinese blocks: Perform G2P (ensuring polyphone accuracy), align, and selectively replace.
        4. For Non-Chinese blocks: Keep as is.
        """
        # 1. Dictionary replacement
        pre_segments = self._tokenize_by_replace_dict(text)
        final_output = []

        for content, is_replaced in pre_segments:
            if is_replaced:
                # Content replaced by dictionary, keep as is
                final_output.append(content)
                continue

            # 2. Split into Chinese / Non-Chinese chunks
            chunks = self._split_mixed_text(content)

            for chunk_text, is_zh in chunks:
                if not is_zh:
                    # Non-Chinese (English, Numbers), keep as is
                    final_output.append(chunk_text)
                else:
                    # Chinese block: Feed entire block to G2P for accurate context
                    # process_one returns flat list: ['n', '|', 'i', '-', 'h', '|', 'ao']
                    try:
                        full_phonemes = process_one(chunk_text, self.text_tokenizer)
                        # Alignment and replacement logic
                        aligned_text = self._align_and_replace(chunk_text, full_phonemes)
                        final_output.append(aligned_text)
                    except Exception as e:
                        print(f"G2P Error for chunk {chunk_text}: {e}")
                        final_output.append(chunk_text)  # Fallback

        return "".join(final_output)

    def replace_with_prob(self, text: str, prob: float = 0.2, max_ratio: float = 0.5) -> str:
        """
        Randomly replaces Chinese characters with their phonemes.
        
        Args:
            text (str): Input text.
            prob (float): Probability to trigger the replacement process (0.0 to 1.0).
            max_ratio (float): Maximum ratio of characters to replace if triggered.
        """
        # 1. Convert text to list
        text_list = list(text)
        phoneme_list = []
        candidate_indices = []  # Record indices that CAN be replaced (Chinese chars)

        # 2. Pre-calculate phonemes for all candidates
        for i, content in enumerate(text_list):
            if content in self.able_list:
                try:
                    phones = process_one(content, self.text_tokenizer)
                    # Formatting: ['sh', '|', 'ang'] -> <|SH|><|ANG|>
                    phones_formatted = [p.upper().replace("-", " ") for p in phones]
                    # Tag phonemes with <||>, exclude punctuation
                    phones_tagged = ["<|" + p + "|>" if p not in PUNCTUATION_CHARS else p for p in phones_formatted]
                    phoneme_str = "".join(phones_tagged)

                    phoneme_list.append(phoneme_str)
                    candidate_indices.append(i)
                except Exception:
                    phoneme_list.append(None)
            else:
                phoneme_list.append(None)

        num_candidates = len(candidate_indices)

        # 3. Determine whether to replace (controlled by prob)
        if random.random() < prob:
            # 4. Determine the number of replacements (controlled by max_ratio)
            max_replace_count = int(num_candidates * max_ratio)
            num_to_replace = random.randint(0, max_replace_count)
            num_to_replace = max(0, min(num_to_replace, num_candidates))
            
            if num_to_replace > 0:
                indices_to_replace = random.sample(candidate_indices, num_to_replace)
                for idx in indices_to_replace:
                    if phoneme_list[idx]:
                        text_list[idx] = phoneme_list[idx]

        return "".join(text_list)


class TTSFrontEnd:
    """
    Unified Frontend for TTS, managing Text Frontend, Speech Tokenizer, and Speaker Embedding extraction.
    """
    LOWER_SR = 16000
    #HIGH_SR = 22050
    HIGH_SR = 24000

    def __init__(self,
                 tokenize_fn: Callable,
                 speech_tokenizer: SpeechTokenizer,
                 feat_extractor: Callable,
                 campplus_model: str,
                 spk2info: str = '',
                 device=None):

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.tokenize_fn = tokenize_fn
        self.feat_extractor = feat_extractor

        # Initialize ONNX models (speaker embed & speech token)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        
        # Determine providers based on availability
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
             providers.insert(0, ('CUDAExecutionProvider', {
                 'device_id': 0,
                 'arena_extend_strategy': 'kSameAsRequested',
                 'cudnn_conv_algo_search': 'DEFAULT',
             }))

        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=providers)
        self.speech_tokenizer = speech_tokenizer

        # Load speaker info if available
        if os.path.exists(spk2info):
            self.spk2info = torch.load(spk2info, map_location=self.device)

    def _extract_text_token(self, text: str) -> torch.Tensor:
        text_token = self.tokenize_fn(text)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        return text_token

    def _extract_speech_token(self, path_or_tuple):
        prompt_speech_tokens = self.speech_tokenizer.extract_speech_token(path_or_tuple)
        return torch.tensor(prompt_speech_tokens).to(self.device)

    def _extract_spk_embedding(self, speech: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(speech, str):
            speech = load_wav(speech, 16000)

        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        
        # ONNX Inference
        input_name = self.campplus_session.get_inputs()[0].name
        embedding = self.campplus_session.run(None, {input_name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech, sample_rate=24000):
        if isinstance(speech, str):
            speech = load_wav(speech, sample_rate)
        speech = speech.to(self.device)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        return speech_feat

    def postprocess(self,speech, top_db=60, hop_length=220, win_length=440):
        max_val = 0.8
  
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
  
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
  
        zeros = torch.zeros(1, int(self.HIGH_SR * 0.2))
  
        print(speech, zeros)
  
        speech = torch.concat([speech, zeros], dim=1)

        return speech
  
    def load_spk_from_wav(self, wav_file):
        target_wav, sample_rate = torchaudio.load(wav_file)
        if target_wav.shape[0] == 2:
            # 计算两个声道的平均值
            target_wav = target_wav.mean(dim=0, keepdim=True)
  
        target_wav_high = torchaudio.transforms.Resample(sample_rate, self.HIGH_SR)(target_wav)
        target_wav_high = self.postprocess(target_wav_high)
        target_wav_lower = torchaudio.transforms.Resample(self.HIGH_SR, self.LOWER_SR)(target_wav_high)
  
        speech_feat = self._extract_speech_feat(target_wav_high)
        speech_token = self._extract_speech_token(target_wav_lower)
        embedding = self._extract_spk_embedding(target_wav_lower)
  
        print(f"speech_feat {type(speech_feat)}")
        print(f"speech_token {type(speech_token)}")    
        print(f"embedding {type(embedding)}")
  
        return {
            "speech_feat": speech_feat,
            "speech_token": speech_token,
            "embedding": embedding
        }

def test_text_frontend():
    # Test example
    frontend = TextFrontEnd(use_phoneme=True)
    text = frontend.text_normalize("You're absolutely killing it! Keep that amazing energy up—nothing can stop you, girl! You're gonna rock it!")
    print(f"English Normalization: {text}")
    
    text = frontend.text_normalize("噢，我知道了。")
    print(f"Chinese Normalization: {text}")

    text = "你好，世界！This is a test sentence with numbers 123 and symbols #@$%."
    normalized_text = frontend.text_normalize(text)
    print(f"Normalized Text: {normalized_text}")

    g2p_text = frontend.g2p_infer(normalized_text)
    print(f"G2P Text: {g2p_text}")

def test_speech_frontend():
    pass


"""
python -m cosyvoice.cli.frontend
"""
if __name__ == "__main__":
    test_text_frontend()