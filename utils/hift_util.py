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

import torch
import pathlib
from typing import Union
from utils.audio import mel_spectrogram
from cosyvoice.hifigan_cosy2.f0_predictor import ConvRNNF0Predictor
from cosyvoice.hifigan_cosy2.generator import HiFTGenerator

class HiFTInference:
    def __init__(self, 
                 ckpt_path: Union[str, pathlib.Path], 
                 device: Union[str, torch.device] = 'cuda',
                 load_only_nsf: bool = False):
        """
        Wrapper class for HiFTGenerator inference.
        """
        if isinstance(ckpt_path, str):
            ckpt_path = pathlib.Path(ckpt_path)
        
        self.device = device
        self.sample_rate = 24000
        
        self.model = self._build_model()
        
        self._load_weights(ckpt_path, load_only_nsf)
        
        self.model.eval()

    def _build_model(self) -> HiFTGenerator:
        """Instantiate the F0 Predictor and HiFTGenerator."""
        f0_predictor = ConvRNNF0Predictor(
            num_class=1,
            in_channels=80,
            cond_channels=512
        )

        # Instantiate HiFTGenerator with CosyVoice 2.0 parameters
        hift = HiFTGenerator(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=self.sample_rate,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_params={
                "n_fft": 16,
                "hop_len": 4
            },
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
            f0_predictor=f0_predictor
        )
        return hift.to(self.device)

    def _load_weights(self, ckpt_path: pathlib.Path, load_only_nsf: bool):
        """Load state dictionary into the model."""
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
        # Clear CUDA cache to free up memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state_dict = torch.load(ckpt_path, map_location=self.device)

        if not load_only_nsf:
            self.model.load_state_dict(state_dict)
        else:
            # Load only specific parts: m_source and f0_predictor
            m_source_dict = {}
            f0_predictor_dict = {}
            for key, value in state_dict.items():
                if 'm_source' in key:
                    m_source_dict[key.replace('m_source.', '')] = value
                if 'f0_predictor' in key:
                    f0_predictor_dict[key.replace('f0_predictor.', '')] = value

            self.model.m_source.load_state_dict(m_source_dict, strict=True)
            self.model.f0_predictor.load_state_dict(f0_predictor_dict, strict=True)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Inference entry point.
        Input: Mel Spectrogram (B, C, T)
        Output: Audio Waveform (B, 1, T_out)
        """
        mel = mel.to(self.device)
        with torch.no_grad():
            audio, _ = self.model.inference(mel)
        return audio

    @staticmethod
    def extract_mel(wav: torch.Tensor) -> torch.Tensor:
        """
        Static method to extract mel spectrogram consistent with training.
        """
        return mel_spectrogram(
            wav,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False
        )

def load_hift(device: str = "cuda", load_only_nsf: bool = False, ckpt_path: str = '') -> HiFTInference:
    """Factory function to load HiFT model."""
    # Update this path to your actual relative path for the open source release
    ckpt_path = ckpt_path or 'ckpt/hift/hift.pt'
    print(f"Loading HiFT model from {ckpt_path} on {device}...")
    return HiFTInference(ckpt_path, device=device, load_only_nsf=load_only_nsf)