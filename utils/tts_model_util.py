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
import queue
from typing import List, Tuple, Generator, Optional, Union

import torch
import numpy as np

from utils.vocos_util import load_vocos_jit
from utils.hift_util import load_hift

class Token2Wav:
    def __init__(self, flow, sample_rate: int = 24000, device: str = "cuda",ckpt_path: str = ""):
        self.device = device
        self.flow = flow
        self.input_frame_rate = flow.input_frame_rate

        # Assume Flow model frame rate is 50Hz
        # corresponding to hop_size 640 at 32k (32000/50 = 640)
        if sample_rate == 32000:
            self.hop_size = 640
            self.sample_rate = 32000
            self.vocoder = load_vocos_jit(device, ckpt_path)
        elif sample_rate == 24000:
            self.hop_size = 480
            self.sample_rate = 24000
            self.vocoder = load_hift(device, ckpt_path=ckpt_path)
        else:
            raise ValueError(f"Unsupported sample_rate: {sample_rate}")
    
    def token2wav_stream(self,
                     syn_token: List[int],
                     block_sizes: List[int] = [25, 50, 200],
                     look_future_sizes: List[int] = [12, 25, 100],
                     fade_sec: float = 0.1,
                     embedding: Optional[torch.Tensor] = None,
                     prompt_token_list: Optional[torch.Tensor] = None,
                     prompt_feat_td: Optional[torch.Tensor] = None,
                     n_timesteps: int = 10,
                     queue: queue.Queue = None,
                     ) -> Tuple[torch.Tensor, List[float], List[float], List[np.ndarray], torch.Tensor]:
        
        if not isinstance(syn_token, list):
            raise TypeError("syn_token must be a list.")
        assert len(block_sizes) == len(look_future_sizes), "block_sizes and look_future_sizes must have the same length."
        if prompt_token_list is None or len(prompt_token_list) == 0:
             raise ValueError("prompt_token_list cannot be empty.")

        diff_cache = None
        result_wav_list = []
        wav_len_pointer = 0
        last_fade_out_array = None

        # Split syn_token into substrings based on block_sizes to simulate streaming input
        chunked_list = list(self.iterate_list_in_chunks(syn_token, block_sizes))
        all_patch_token = []
        mel_list = []

        for i, patch_token in enumerate(chunked_list):
            all_patch_token += patch_token
            
            # Inference with cache (Flow matching / Diffusion)
            # block_pattern guides the transformer attention mask creation
            # Note: prompt_token length is added to the beginning as we introduced prompt tokens
            mel_bdt, diff_cache = self.flow.inference_with_cache(
                token=torch.LongTensor(all_patch_token)[None].to(self.device),
                prompt_token=prompt_token_list.to(self.device),
                prompt_feat=prompt_feat_td.to(self.device),
                embedding=embedding.to(self.device),
                last_step_cache=diff_cache,
                is_causal=True,
                block_pattern=[len(prompt_token_list)] + block_sizes
            )

            # [Modification] Replace with Vocos inference, return wav tensor directly
            wav_bt = self.vocoder(mel_bdt)
            if queue is not None:
                queue.put_nowait((wav_bt, mel_bdt))
            
            mel_list.append(mel_bdt)
            wav_npy = wav_bt.squeeze().detach().cpu().numpy()

            # Subsequent logic relies mainly on self.sample_rate calculation, no major changes needed
            min_index = min(len(look_future_sizes) - 1, i)
            look_future_sec = look_future_sizes[min_index] / self.input_frame_rate

            # Total WAV length calculation
            total_sec = len(wav_npy) / self.sample_rate
            
            # "Total length - look_future" is the length we need to override/update in the next inference
            override_sec = total_sec - look_future_sec
            override_mel_len = int(override_sec * self.sample_rate / self.hop_size)
            
            # Update cache for the next step
            diff_cache['override_len'] = override_mel_len + 1
            if prompt_feat_td is not None:
                prompt_mel_len = prompt_feat_td.shape[0]
                diff_cache['override_len'] = prompt_mel_len + override_mel_len + 1

            # Calculate overlap and look-back lengths
            overlap_sec = look_future_sec + fade_sec
            overlap_len = int(self.sample_rate * overlap_sec)
            look_back_len = int(self.sample_rate * look_future_sec)

            # --- Stream Stitching Logic ---
            
            # Case 1: First chunk
            if i == 0:
                if len(chunked_list) == 1:
                    result_wav_list.append(wav_npy)
                    continue
                
                # 1. Non-overlap area, safe to return/play
                # Ensure we don't slice with negative index if wav is too short
                valid_len = max(0, len(wav_npy) - overlap_len)
                result_wav_list.append(wav_npy[:valid_len])
                wav_len_pointer += len(result_wav_list[-1])

                # 2. Fade area, stored for next iteration
                last_fade_out_array = wav_npy[-overlap_len:-look_back_len] if look_back_len > 0 else wav_npy[-overlap_len:]
                continue

            # Case 2: Subsequent chunks
            # 1. Remove the head part that was already generated in the previous step
            current_wav = wav_npy[wav_len_pointer:]

            # 2. Apply cross-fade at the boundary
            if last_fade_out_array is not None:
                # FIX: Handle case where current_wav is shorter than the fade array
                fade_len = min(len(last_fade_out_array), len(current_wav))
                fade_in_array = np.linspace(0, 1, fade_len)
                fade_out_array = np.linspace(1, 0, fade_len)
                
                current_wav[0:fade_len] = (
                    last_fade_out_array[:fade_len] * fade_out_array + 
                    current_wav[0:fade_len] * fade_in_array
                )

            # Case 3: Last chunk
            if i == len(chunked_list) - 1:
                result_wav_list.append(current_wav)
                break

            # 3. Return content (minus the overlap for the next chunk)
            valid_len = max(0, len(current_wav) - overlap_len)
            result_wav_list.append(current_wav[:valid_len])
            wav_len_pointer += len(result_wav_list[-1])

            # 4. Update fade area for the next iteration
            last_fade_out_array = current_wav[-overlap_len:-look_back_len] if look_back_len > 0 else current_wav[-overlap_len:]

        # Statistics: length of each segment
        sec_list = [len(wav) / self.sample_rate for wav in result_wav_list]
        
        if result_wav_list:
            wav_full = np.concatenate(result_wav_list)
            wav_bt = torch.FloatTensor(wav_full)[None]
        else:
            wav_bt = torch.zeros(1, 0)

        # --- Consistency Check (Optional debugging info) ---
        diff_list = []
        for i in range(len(mel_list) - 1):
            mel_small = mel_list[i]
            mel_big = mel_list[i + 1]

            # 1. Truncate the tail where tokens were insufficient
            look_future_size = look_future_sizes[min(i, len(look_future_sizes) - 1)]
            look_future_mel_len = int(look_future_size / self.input_frame_rate * self.sample_rate / self.hop_size)
            
            if mel_small.shape[-1] > look_future_mel_len:
                mel_small = mel_small[:, :, 0:mel_small.shape[-1] - look_future_mel_len]
            
            # Align shapes
            current_len = mel_small.shape[-1]
            if mel_big.shape[-1] >= current_len:
                mel_big = mel_big[:, :, 0:current_len]
            else:
                # Skip comparison if shapes don't align unexpectedly
                continue

            # 2. Only compare the fade area
            overlap_mel_len = int(fade_sec * self.sample_rate / self.hop_size)
            if current_len >= overlap_mel_len:
                mel_small = mel_small[:, :, -overlap_mel_len:]
                mel_big = mel_big[:, :, -overlap_mel_len:]
                
                diff = self.calc_ratio(mel_small, mel_big) * 100
                print(f"Chunk {i}: diff:{diff :.2f}%") # Optional logging
                diff_list.append(diff)

        return wav_bt, sec_list, diff_list, result_wav_list, mel_list
    
    def token2wav_with_cache(self,
                             token_bt: Union[List[int], np.ndarray, torch.Tensor],
                             n_timesteps: int = 10,
                             prompt_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
                             prompt_feat: torch.Tensor = torch.zeros(1, 0, 80),
                             embedding: torch.Tensor = torch.zeros(1, 192),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(token_bt, (list, np.ndarray)):
            token_bt = torch.tensor(token_bt, dtype=torch.long)[None]
        elif not isinstance(token_bt, torch.Tensor):
             raise ValueError(f"Unsupported token_bt type: {type(token_bt)}")

        assert prompt_token.shape[1] != 0 and prompt_feat.shape[1] != 0
        mel, _ = self.flow.inference_with_cache(
            token=token_bt.to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            embedding=embedding.to(self.device),
            n_timesteps=n_timesteps,
        )
        
        wav = self.vocoder(mel)

        return wav, mel

    def calc_ratio(self, small_out: torch.Tensor, big_out: torch.Tensor) -> float:
        # Ensure dimensions match before operation to avoid runtime errors
        min_len = min(small_out.shape[-1], big_out.shape[-1])
        diff = (small_out[..., :min_len] - big_out[..., :min_len]).abs()
        
        sum_abs = small_out.abs().sum()
        if sum_abs == 0:
            return 0.0
        
        ratio = (diff.sum() / sum_abs).item()
        return ratio

    def iterate_list_in_chunks(self, lst: List, chunk_sizes: List[int]) -> Generator[List, None, None]:
        if not lst:
            return
            
        pointer = 0
        # Iterate through defined chunk sizes
        for chunk_size in chunk_sizes:
            start = pointer
            end = start + chunk_size
            if start >= len(lst):
                break
            result = lst[start:end]
            pointer = end
            if result:
                yield result

        # Continue with the last chunk size for the remainder of the list
        if chunk_sizes:
            last_chunk_size = chunk_sizes[-1]
            while pointer < len(lst):
                start = pointer
                end = start + last_chunk_size
                result = lst[start:end]
                pointer = end
                if result:
                    yield result
                else:
                    break