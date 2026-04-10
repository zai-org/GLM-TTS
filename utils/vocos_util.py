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
import torch.jit
import pathlib
from typing import Union

FS32K = 32000
MEL_LOGDIFF = -7.847762537473608

class Vocos2DInference:
    def __init__(self, in_ckpt_path: Union[str, pathlib.Path], device: Union[str, torch.device] = 'cpu'):
        # Ensure path is a Path object
        if isinstance(in_ckpt_path, str):
            in_ckpt_path = pathlib.Path(in_ckpt_path)
        
        self.device = device
        # Load JIT model
        self.gen_model: torch.jit.RecursiveScriptModule = torch.jit.load(in_ckpt_path, map_location=device)
        _ = self.gen_model.eval()

    def __call__(self, xs_mel_tr: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, n_mel_filterbins, frame_count) -> (B, C, T)
        Output: (batch_size, sample_count)
        """
        # Logic from notebook: transpose(-1, -2) and add MEL_LOGDIFF
        # This implies the model internally expects (B, T, C) format
        if xs_mel_tr.size(-1) == 1:
            xs_mel_tr = torch.cat([xs_mel_tr, xs_mel_tr], dim=-1)
        xs_mel: torch.Tensor = xs_mel_tr.transpose(-1, -2) + MEL_LOGDIFF
        xs:     torch.Tensor = self.gen_model(xs_mel)   # (batch_size, sample_count)
        return xs
    
    def stft_mel(self, xs: torch.Tensor) -> torch.Tensor:
        # Helper function to reverse Mel from audio (contained in Notebook)
        # Not strictly needed for inference but kept for completeness
        xs_spec:   torch.Tensor = self.gen_model._stft(xs)
        xs_mag:    torch.Tensor = xs_spec.abs()
        xs_expmel: torch.Tensor = xs_mag.matmul(self.gen_model.fbs)
        xs_mel:    torch.Tensor = xs_expmel.clamp_min(self.gen_model.spec_min).log()
        xs_mel_tr: torch.Tensor = xs_mel.transpose(-1, -2) - MEL_LOGDIFF
        return xs_mel_tr

def load_vocos_jit(device: str = "cuda", ckpt_path: str = "") -> Vocos2DInference:
    """Factory function to load Vocos model"""
    ckpt_path = ckpt_path or 'ckpt/vocos2d/generator_jit.ckpt'
    print(f"Loading Vocos JIT model from {ckpt_path} on {device}...")
    return Vocos2DInference(ckpt_path, device=device)