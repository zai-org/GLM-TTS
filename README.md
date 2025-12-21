# GLM-TTS: Controllable & Emotion-Expressive Zero-shot TTS with Multi-Reward Reinforcement Learning

[中文阅读](README_zh.md)

<div align="center">
<img src=assets/images/logo.svg  width="50%"/>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2512.14291" target="_blank">📜 Paper</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://huggingface.co/zai-org/GLM-TTS" target="_blank">🤗 HuggingFace</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://modelscope.cn/models/ZhipuAI/GLM-TTS" target="_blank">🤖 ModelScope</a>
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://audio.z.ai/" target="_blank"> 🛠️Audio.Z.AI</a>
</p>

## Model Introduction
GLM-TTS is a high-quality text-to-speech (TTS) synthesis system based on large language models, supporting zero-shot voice cloning and streaming inference. This system adopts a two-stage architecture: first, it uses LLM to generate speech token sequences, then uses Flow model to convert tokens into high-quality audio waveforms. By introducing a Multi-Reward Reinforcement Learning framework, GLM-TTS can generate more expressive and emotional speech, significantly improving the expressiveness of traditional TTS systems.

## News & Updates

- **[2025.12.11]** 🎉 The project is officially open-sourced, featuring inference scripts and a series of model weights.
- **[2025.12.17]** GLM-TTS Technical Report is available on arXiv: [2512.14291](https://arxiv.org/abs/2512.14291).
- **[Coming Soon]** 2D Vocos vocoder update in progress.
- **[Coming Soon]** Model Weights Optimized via Reinforcement Learning

## Features

- **Zero-shot Voice Cloning**: Clone any speaker's voice with just 3-10 seconds of prompt audio
- **RL-enhanced Emotion Control**: Achieve more natural emotional expression and prosody control through multi-reward reinforcement learning framework
- **Streaming Inference**: Support real-time streaming audio generation, suitable for interactive applications
- **High-quality Synthesis**: Generate natural and expressive speech with quality comparable to commercial systems
- **Multi-language Support**: Primarily supports Chinese, while also supporting English mixed text
- **Phoneme-level Modeling**: Support phoneme-level text-to-speech conversion
- **Flexible Inference Methods**: Support multiple sampling strategies and inference modes

## Quick Start

### Environment Setup

Ensure you use Python 3.10 - Python 3.12 versions.

```bash
# Clone repository
git clone https://github.com/zai-org/GLM-TTS.git
cd GLM-TTS

# Install dependencies
pip install -r requirements.txt

# Install reinforcement learning related dependencies (optional)
cd grpo/modules
git clone https://github.com/s3prl/s3prl
git clone https://github.com/omine-me/LaughterSegmentation
# Download wavlm_large_finetune.pth and place it in grpo/ckpt directory
```

### Download Pre-trained Models

We support downloading the complete model weights (including Tokenizer, LLM, Flow, Vocoder, and Frontend) from HuggingFace or ModelScope.

```bash
# Create model directory
mkdir -p ckpt

# Option 1: Download from HuggingFace
pip install -U huggingface_hub
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt

# Option 2: Download from ModelScope
pip install -U modelscope
modelscope download --model ZhipuAI/GLM-TTS --local_dir ckpt
```

### Running Inference Demo

#### Command Line Inference

```bash
python glmtts_inference.py \
    --data=example_zh \
    --exp_name=_test \
    --use_cache \
    # --phoneme # Add this flag to enable phoneme capabilities.
```

#### Shell Script Inference

```bash
bash glmtts_inference.sh
```

#### Interactive Web Interface

```bash
python -m tools.gradio_app
```

## System Architecture

### Overview


GLM-TTS adopts a two-stage design: in the first stage, a large language model (LLM) based on Llama architecture converts input text into speech token sequences; in the second stage, the Flow Matching model converts these token sequences into high-quality mel-spectrogram, and finally generates audio waveforms through a vocoder. The system supports zero-shot voice cloning by extracting speaker features from prompt audio without fine-tuning for specific speakers.

<div align="center">
  <img src="assets/images/architecture.png" width="50%" alt="GLM-TTS Architecture" title="GLM-TTS Architecture">
</div>

### Fine-grained Pronunciation Control (Phoneme-in)

For scenarios demanding high pronunciation accuracy, such as educational assessments and audiobooks, GLM-TTS introduces the **Phoneme-in** mechanism to address automatic pronunciation ambiguity in polyphones (e.g., "行" which can be read as *xíng* or *háng*) and rare characters. This mechanism supports **"Hybrid Phoneme + Text"** input, enabling precise, targeted control over specific vocabulary pronunciation.

- **Hybrid Training**
  During training, random G2P (Grapheme-to-Phoneme) conversion is applied to parts of the text. This strategy compels the model to adapt to hybrid input sequences, preserving its ability to understand pure text while enhancing generalization for phoneme inputs.

- **Targeted Inference**
  Inference follows a `G2P -> Table Lookup Replacement -> Hybrid Input` workflow:
  1. **Global Conversion**: Obtain the complete phoneme sequence for the input text.
  2. **Dynamic Replacement**: Using a "Dynamic Controllable Dictionary," automatically identify polyphones or rare characters and replace them with specified target phonemes.
  3. **Hybrid Generation**: Feed the combination of replaced phonemes and original text into GLM-TTS as a hybrid input. This ensures precise pronunciation control for specific words while maintaining natural prosody.


### RL Alignment

<div align="center">
  <img src="assets/images/rl.png" width="70%" alt="GLM-TTS RL" title="GLM-TTS RL">
</div>

To address the issue of flat emotional expression in traditional TTS, we introduce a multi-reward reinforcement learning framework. This framework comprehensively evaluates generated speech through multiple reward functions (including similarity reward, CER reward, emotion reward, laughter reward, etc.) and uses the GRPO (Group Relative Policy Optimization) algorithm to optimize the LLM's generation strategy. Specifically:

1. **Multi-reward Design**: The system designs various reward functions to evaluate the quality of generated speech from different dimensions, including sound quality, similarity, emotional expression, etc.
2. **Reward Server**: Computes multiple reward functions through a distributed reward server, supporting parallel processing
3. **Policy Optimization**: Uses GRPO algorithm to optimize the LLM's generation strategy based on reward signals, enhancing the emotional expressiveness of speech
4. **Token-level Rewards**: Supports fine-grained token-level reward allocation, providing more precise optimization signals

Through RL optimization, GLM-TTS_RL reduces the CER metric from 1.03 to 0.89 compared to the base model, while maintaining high similarity, achieving better sound quality and expressiveness.

## Core Components & Implementation

### LLM Backend
- **File Location**: [`llm/glmtts.py`](llm/glmtts.py)
- **Function**: Text-to-speech model based on Llama architecture, responsible for converting input text into speech token sequences
- **Supported Modes**: Pretrained (PRETRAIN), Fine-tuning (SFT), and LoRA modes

### Flow Matching
- **File Location**: [`flow/`](flow/) directory
- **Core Files**: 
  - [`dit.py`](flow/dit.py): Diffusion Transformer implementation, supporting conditional generation
  - [`flow.py`](flow/flow.py): Streaming inference support, implementing real-time audio generation
- **Function**: Converts token sequences generated by LLM into high-quality mel-spectrogram

### Frontend
- **File Location**: [`cosyvoice/cli/frontend.py`](cosyvoice/cli/frontend.py)
- **Function**: Preprocessing of text and speech, including text normalization, phoneme conversion, speech token extraction, and speaker embedding extraction
- **Features**: Supports Chinese and English mixed text processing

### Reinforcement Learning Module
- **File Location**: [`grpo/`](grpo/) directory
- **Core Files**:
  - [`grpo_utils.py`](grpo/grpo_utils.py): GRPO algorithm implementation and batch inference
  - [`reward_func.py`](grpo/reward_func.py): Multi-reward function implementation
  - [`reward_server.py`](grpo/reward_server.py): Distributed reward server
- **Function**: Optimizes the emotional expressiveness of the TTS system through multi-reward reinforcement learning

## Evaluation Results

Evaluated on `seed-tts-eval zh testset`. To maintain consistency with the original evaluation, inference was performed without the `--phoneme` flag.

**CER**: Character Error Rate (lower is better $\downarrow$) | **SIM**: Similarity (higher is better $\uparrow$)

| Model | CER $\downarrow$ | SIM $\uparrow$ | Open-source |
| :--- | :---: | :---: | :---: |
| MegaTTS3 | 1.52 | 79.0 | 🔒 No |
| DiTAR | 1.02 | 75.3 | 🔒 No |
| CosyVoice3 | 1.12 | 78.1 | 🔒 No |
| Seed-TTS | 1.12 | **79.6** | 🔒 No |
| MiniMax | **0.83** | 78.3 | 🔒 No |
| CosyVoice2 | 1.38 | 75.7 | 👐 Yes |
| F5-TTS | 1.53 | 76.0 | 👐 Yes |
| FireRedTTS-2 | 1.14 | 73.6 | 👐 Yes |
| IndexTTS2 | 1.03 | 76.5 | 👐 Yes |
| VibeVoice | 1.16 | 74.4 | 👐 Yes |
| HiggsAudio-v2 | 1.50 | 74.0 | 👐 Yes |
| VoxCPM | 0.93 | 77.2 | 👐 Yes |
| **GLM-TTS (Ours)** | 1.03 | 76.1 | 👐 Yes |
| **GLM-TTS_RL (Ours)** | **0.89** | 76.4 | 👐 Yes |

## Project Structure

```
GLM-TTS/
├── glmtts_inference.py              # Main inference script, containing complete inference process
├── glmtts_inference.sh              # Pre-trained model inference script
├── configs/                         # Configuration files directory
│   ├── spk_prompt_dict.yaml         # Speaker prompt dictionary
│   ├── lora_adapter_configV3.1.json # LoRA adapter configuration
│   ├── G2P_able_1word.json          # Single character phoneme conversion configuration
│   ├── G2P_all_phonemes.json        # Full phoneme list
│   ├── G2P_replace_dict.jsonl       # Phoneme replacement dictionary
│   └── custom_replace.jsonl         # Custom replacement rules
├── cosyvoice/                       # Cosyvoice module
│   ├── cli/
│   │   └── frontend.py              # Text and speech frontend processing
│   └── utils/                       # Utility functions
├── examples/                        # Example data
│   ├── *.jsonl                      # Example jsonl files
│   └── prompt/                      # Prompt audio directory
│       ├── *.wav                    # Prompt audio (for research use only)
│       └── LICENSE                  # Audio file license
├── flow/                            # Flow model related
│   ├── dit.py                       # Diffusion Transformer implementation
│   ├── flow.py                      # Streaming Flow model
│   └── modules.py                   # Flow model basic modules
├── grpo/                            # Reinforcement learning module
│   ├── grpo_utils.py                # GRPO algorithm implementation
│   ├── reward_func.py               # Multi-reward functions
│   ├── reward_server.py             # Distributed reward server
│   ├── train_ds_grpo.py             # GRPO training script
│   └── data/                        # Training data and configuration
├── llm/                             # Large language model related
│   └── glmtts.py                    # GLM-TTS LLM implementation
├── frontend/                        # Frontend model files
│   ├── campplus.onnx                # Speaker embedding model
│   └── cosyvoice_frontend.yaml      # Frontend configuration
├── tools/                           # Tool scripts
│   ├── gradio_app.py                # Gradio interactive interface
│   ├── ffmpeg_speech_control.py     # Audio processing tool
│   └── flow_reconstruct.py          # Audio reconstruction
└── utils/                           # Common utilities
    ├── tts_model_util.py            # TTS model utilities
    ├── yaml_util.py                 # YAML configuration loading utility
    ├── audio.py                     # Audio processing utility
    ├── seed_util.py                 # Random seed utility
    ├── block_mask_util.py           # Block mask utility
    ├── vocos_util.py                # Vocos vocoder utility
    ├── hift_util.py                 # Hift vocoder utility
    ├── whisper_models/              # Whisper model components
    └── glm_g2p.py                   # Text to phoneme conversion
```

## Acknowledgments

We thank the following open-source projects for their support:

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - Providing frontend processing framework and high-quality vocoder
- [Llama](https://github.com/meta-llama/llama) - Providing basic language model architecture
- [Vocos](https://github.com/charactr-platform/vocos) - Providing high-quality vocoder
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) - Reinforcement learning algorithm implementation inspiration

---
## Citation

If you find GLM-TTS useful for your research, please cite our technical report:

```bibtex
@misc{cui2025glmttstechnicalreport,
      title={GLM-TTS Technical Report}, 
      author={Jiayan Cui and Zhihan Yang and Naihan Li and Jiankun Tian and Xingyu Ma and Yi Zhang and Guangyu Chen and Runxuan Yang and Yuqing Cheng and Yizhi Zhou and Guochen Yu and Xiaotao Gu and Jie Tang},
      year={2025},
      eprint={2512.14291},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2512.14291}, 
}