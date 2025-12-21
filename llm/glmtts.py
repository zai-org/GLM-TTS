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
import json
import yaml
import queue
from typing import Union, Optional, List, Dict, Any
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from cosyvoice.utils import common


class GLMTTS(nn.Module):
    """
    GLMTTS model based on Llama architecture for Text-to-Speech tasks.
    Supports Pretraining, SFT, and LoRA fine-tuning modes.
    """

    def __init__(
        self,
        llama_cfg_path: str,
        mode: str = "PRETRAIN",
        lora_adapter_config: Optional[str] = 'configs/lora_adapter_configV3.1.json',
        spk_prompt_dict_path: Optional[str] = 'configs/spk_prompt_dict.yaml',
    ):
        """
        Initialize the GLMTTS model.

        Args:
            llama_cfg_path (str): Path to the Llama configuration JSON file.
            mode (str): Training/Inference mode. Options: "PRETRAIN", "SFT", "LORA".
            lora_adapter_config (Optional[str]): Path to the LoRA adapter config file.
            spk_prompt_dict_path (Optional[str]): Path to the speaker prompt dictionary (YAML).
        """
        super().__init__()
        
        # Load speaker prompt dictionary if provided
        self.spk_prompt_dict: Optional[Dict[str, Any]] = None
        if spk_prompt_dict_path:
            with open(spk_prompt_dict_path, "r", encoding="utf-8") as f:
                self.spk_prompt_dict = yaml.safe_load(f)

        self.special_token_ids: Optional[Dict[str, int]] = None
        self.ats: Optional[int] = None  # Audio Token Start
        self.ate: Optional[int] = None  # Audio Token End
        self.boa: Optional[int] = None  # Beginning of Audio
        self.eoa: Optional[int] = None  # End of Audio
        self.pad: Optional[int] = None  # Padding Token

        self.mode = mode
        
        # Initialize Llama model
        config = LlamaConfig.from_json_file(llama_cfg_path)
        self.llama = LlamaForCausalLM(config)
        self.llama_embedding = self.llama.model.embed_tokens
        self.lora_adapter_config = lora_adapter_config

    def apply_lora(self) -> None:
        """
        Apply LoRA adapter to the Llama model based on the configuration file.
        """
        if not hasattr(self, 'lora_adapter_config') or not self.lora_adapter_config:
            print("Warning: LoRA config not found, skipping apply_lora.")
            return

        with open(self.lora_adapter_config, 'r') as f:
            adapter_config = json.load(f)

        lora_config = LoraConfig(
            r=adapter_config['r'],
            lora_alpha=adapter_config['lora_alpha'],
            target_modules=adapter_config['target_modules'],
            lora_dropout=adapter_config['lora_dropout'],
            bias="none",
            init_lora_weights=adapter_config['init_lora_weights'],
            modules_to_save=adapter_config.get('modules_to_save', None),
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

    def set_runtime_vars(self, special_token_ids: Dict[str, int]) -> None:
        """
        Set special token IDs required for inference.
        
        Args:
            special_token_ids (Dict[str, int]): Dictionary containing keys:
                                              ['ats', 'ate', 'boa', 'eoa', 'pad']
        """
        required_keys = ['ats', 'ate', 'boa', 'eoa', 'pad']
        assert all(k in special_token_ids for k in required_keys), \
            f"special_token_ids must contain all keys: {required_keys}"
            
        self.special_token_ids = special_token_ids
        self.ats = special_token_ids['ats']
        self.ate = special_token_ids['ate']
        self.boa = special_token_ids['boa']
        self.eoa = special_token_ids['eoa']
        self.pad = special_token_ids['pad']

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        sampling: Union[bool, int, float] = True,
        beam_size: int = 1,
        ignore_eos: bool = True,
    ) -> torch.Tensor:
        """
        Perform sampling on weighted scores. 
        Re-samples if EOS is generated when ignore_eos is True.
        """
        while True:
            # Get top-k probabilities and indices
            prob, indices = weighted_scores.softmax(dim=-1).topk(sampling)
            # Multinomial sampling
            top_ids_index = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[top_ids_index]
            
            # If we allow EOS or if EOS was not generated, break the loop
            if (not ignore_eos) or (self.eoa not in top_ids):
                break
        return top_ids

    def sampling_ids_ras(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List[int],
        sampling: int,
    ) -> torch.Tensor:
        """
        Wrapper for RAS (Random Access Sampling) method.
        """
        return common.ras_sampling(weighted_scores, decoded_tokens, sampling, temperature=1)

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        beam_size: int = 1,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        sample_method: str = "ras",
        spk: str = "tongtong",
        queue: queue.Queue = None,
    ) -> torch.Tensor:
        """
        Autoregressive inference loop to generate speech tokens from text.

        Args:
            text: Input text token tensor.
            text_len: Length of input text.
            prompt_text: Prompt text tensor.
            prompt_text_len: Length of prompt text.
            prompt_speech_token: Prompt speech token tensor.
            prompt_speech_token_len: Length of prompt speech tokens.
            beam_size: Beam size for sampling (default 1).
            sampling: Top-k value or sampling parameter.
            max_token_text_ratio: Multiplier to determine max generation length.
            min_token_text_ratio: Multiplier to determine min generation length.
            sample_method: 'ras' or 'topk'.
            spk: Speaker key for SFT mode.

        Returns:
            torch.Tensor: Generated audio tokens (shifted by ATS offset).
        """
        device = text.device

        # 1. Preprocess Prompt Tokens
        # If prompts exist, add the audio start token offset if necessary
        # (Note: Logic depends on how prompt_speech_token is pre-processed outside)
        if prompt_speech_token_len != 0 and prompt_text_len != 0:
            prompt_speech_token = prompt_speech_token + self.ats

        # 2. Construct Input Embeddings
        boa_tensor = torch.tensor([self.boa], device=device).unsqueeze(0)
        
        if self.mode == "SFT":
            if spk not in self.spk_prompt_dict:
                raise ValueError(f"Speaker '{spk}' not found in spk_prompt_dict.")
            
            spk_prompt_tensor = torch.tensor(self.spk_prompt_dict[spk], device=device).unsqueeze(0)
            
            # Concatenate: [Speaker Prompt, Text Prompt, Text, BOA, Speech Prompt]
            input_ids = torch.cat([
                spk_prompt_tensor, 
                prompt_text, 
                text, 
                boa_tensor, 
                prompt_speech_token
            ], dim=1).to(torch.long)
            
            inputs_embeds = self.llama_embedding(input_ids)
            
        elif self.mode in ["PRETRAIN", "LORA"]:
            # Concatenate: [Text Prompt, Text, BOA, Speech Prompt]
            input_ids = torch.cat([
                prompt_text, 
                text, 
                boa_tensor, 
                prompt_speech_token
            ], dim=1).to(torch.long)
            
            inputs_embeds = self.llama_embedding(input_ids)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # 3. Calculate Generation Bounds
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # 4. Step-by-Step Decoding
        out_tokens = []
        past_key_values = None

        for i in range(max_len):
            model_input = {
                "inputs_embeds": inputs_embeds,
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": True,
                "past_key_values": past_key_values
            }

            outputs = self.llama(**model_input)
            past_key_values = outputs['past_key_values']
            
            # Get logits of the last token
            logp = outputs['logits'][:, -1].log_softmax(dim=-1)
                
            # --- Sampling Logic ---
            if sample_method == "ras":
                # Mask the EOS token logit to negative infinity to prevent early stopping
                # if we haven't reached the minimum length.
                if i < min_len:
                    logp[:, self.eoa] = -float('inf')
                
                top_ids = self.sampling_ids_ras(
                    logp.squeeze(dim=0), 
                    out_tokens, 
                    sampling
                ).item()
            elif sample_method == "topk":
                top_ids = self.sampling_ids(
                    logp.squeeze(dim=0), 
                    sampling, 
                    beam_size,
                    ignore_eos=(i < min_len)
                ).item()
            else:
                raise ValueError(f"Unknown sample_method: {sample_method}")

            # Check for End of Audio
            if top_ids == self.eoa:
                break
                
            if queue is not None:
                queue.put_nowait(top_ids)
            out_tokens.append(top_ids)
            
            # Prepare input for the next step (auto-regressive) use cache prefix
            inputs_embeds = self.llama_embedding(torch.LongTensor([top_ids]).to(device))[None]

        # 5. Validation and Output Construction
        # Ensure all tokens are within the valid audio token range
        for token in out_tokens:
            if not (self.ats <= token <= self.ate):
                # Using print for warning instead of crashing in inference
                print(f"Warning: Token {token} is out of the valid range ({self.ats}, {self.ate})")
        
        # Return tokens relative to Audio Token Start (ATS)
        return torch.tensor([out_tokens], dtype=torch.int64, device=device) - self.ats