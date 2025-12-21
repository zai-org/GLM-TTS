# GLM-TTS: Controllable & Emotion-Expressive Zero-shot TTS with Multi-Reward Reinforcement Learning

[Read this in English](README.md)

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

## 模型介绍
GLM-TTS 是一个基于大型语言模型的高质量文本到语音(TTS)合成系统，支持零样本语音克隆和流式推理。本系统采用两阶段架构：首先使用LLM生成语音token序列，然后使用Flow模型将token转换为高质量的音频波形。通过引入多奖励强化学习(Multi-Reward RL)框架，GLM-TTS能够生成更加富有表现力和情感色彩的语音，显著提升了传统TTS系统的表现力。

## 更新计划

- **[2025.12.11]** 🎉 GLM-TTS 项目正式开源！包括推理脚本和一系列模型权重。
- **[2025.12.17]** 📚 GLM-TTS 技术报告已在 arXiv 上发布：[2512.14291](https://arxiv.org/abs/2512.14291)。
- **[Coming Soon]** 2D-Vocos声码器更新中。
- **[Coming Soon]** 强化学习优化后的模型权重。

## 主要特性

- **零样本语音克隆**：仅需3-10秒的提示音频即可克隆任意说话人声音
- **RL-增强的情感控制**：通过多奖励强化学习框架，实现更自然的情感表达和韵律控制
- **流式推理**：支持实时流式音频生成，适用于交互式应用
- **高质量合成**：生成自然、富有表现力的语音，音质媲美商业系统
- **多语言支持**：主要支持中文，同时支持英文混合文本
- **音素级建模**：支持音素级别的文本到语音转换
- **灵活的推理方式**：支持多种采样策略和推理模式

## 快速开始

### 环境安装

确保你使用 Python3.10 - Python3.12 版本。

```bash
# 克隆仓库
git clone https://github.com/zai-org/GLM-TTS.git
cd GLM-TTS

# 安装依赖
pip install -r requirements.txt

# 安装强化学习相关依赖（可选）
cd grpo/modules
git clone https://github.com/s3prl/s3prl
git clone https://github.com/omine-me/LaughterSegmentation
# 下载 wavlm_large_finetune.pth 并放置在 grpo/ckpt 目录
```

### 预训练模型下载

支持从 HuggingFace 或 ModelScope 下载完整模型文件（包含 Tokenizer, LLM, Flow, Vocoder 及 Frontend 模型）。

```bash
# 创建模型目录
mkdir -p ckpt

# 方式一：从 HuggingFace 下载
pip install -U huggingface_hub
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt

# 方式二：从 ModelScope 下载
pip install -U modelscope
modelscope download --model ZhipuAI/GLM-TTS --local_dir ckpt
```

### 运行推理 demo

#### 命令行推理

```bash
python glmtts_inference.py \
    --data=example_zh \
    --exp_name=_test \
    --use_cache \
    # --phoneme # 如果想要使用phoneme能力，请加上这个参数
```

#### 使用shell脚本推理

```bash
bash glmtts_inference.sh
```

#### 交互式Web界面

```bash
python -m tools.gradio_app
```

## 系统架构

### 整体架构

GLM-TTS 采用两阶段设计：第一阶段，基于Llama架构的大型语言模型(LLM)将输入文本转换为语音token序列；第二阶段，Flow Matching模型将这些token序列转换为高质量的梅尔频谱，最后通过声码器生成音频波形。系统支持零样本语音克隆，通过提示音频提取说话人特征，无需针对特定说话人进行微调。

<div align="center">
  <img src="assets/images/architecture.png" width="50%" alt="GLM-TTS Architecture" title="GLM-TTS Architecture">
</div>


### 精细化发音控制 (Phoneme-in)

针对教育评测、有声读物等对发音准确性要求严苛的场景，GLM-TTS 引入了 **Phoneme-in** 机制，旨在解决多音字（例如：”行“ 可以被读成 *xíng* / *háng* ）和生僻字的自动发音歧义问题。该机制允许模型接受 **"Hybrid Phoneme + Text" (混合音素文本)** 输入，实现对特定词汇发音的定向干预。

- **混合模态训练 (Hybrid Training)**
  在训练阶段，对文本中的部分字词随机进行 G2P (Grapheme-to-Phoneme) 转换。这种策略迫使模型适应混合输入序列，既保留了对纯文本的理解能力，又增强了对音素输入的泛化适配性。

- **定向推理 (Targeted Inference)**
  推理时采用 `G2P -> 查表替换 -> 混合输入` 的流程：
  1. **全局转换**：首先获取待合成文本的完整音素序列。
  2. **动态替换**：基于“动态可控词典”，自动识别多音字或生僻字，将其替换为指定的目标音素。
  3. **混合生成**：将替换后的音素与原文未替换部分组合，以混合形式输入 GLM-TTS。这确保了在精准控制特定字词发音的同时，依然保持自然流畅的韵律表现。

### 强化学习优化

<div align="center">
  <img src="assets/images/rl.png" width="70%" alt="GLM-TTS RL" title="GLM-TTS RL">
</div>

为了解决传统 TTS 情感表达平淡的问题，我们引入了多奖励强化学习框架。该框架通过多个奖励函数（包括相似度奖励、CER奖励、情感奖励、笑声奖励等）对生成的语音进行综合评价，并使用GRPO(Group Relative Policy Optimization)算法优化LLM的生成策略。具体而言：

1. **多奖励设计**：系统设计了多种奖励函数，从不同维度评价生成语音的质量，包括音质、相似度、情感表达等
2. **奖励服务器**：通过分布式奖励服务器计算多个奖励函数，支持并行处理
3. **策略优化**：使用GRPO算法，基于奖励信号优化LLM的生成策略，提升语音的情感表现力
4. **Token级奖励**：支持token级别的细粒度奖励分配，提供更精确的优化信号

通过RL优化，GLM-TTS_RL相比基础模型在CER指标上从1.03降低到0.89，同时保持了较高的相似度，实现了更好的音质和表现力。

## 核心组件与实现

### LLM Backend
- **文件位置**: [`llm/glmtts.py`](llm/glmtts.py)
- **功能**: 基于Llama架构的文本到语音模型，负责将输入文本转换为语音token序列
- **支持模式**: 预训练(PRETRAIN)、微调(SFT)和LoRA三种模式

### Flow Matching
- **文件位置**: [`flow/`](flow/)目录
- **核心文件**: 
  - [`dit.py`](flow/dit.py): Diffusion Transformer实现，支持条件生成
  - [`flow.py`](flow/flow.py): 流式推理支持，实现实时音频生成
- **功能**: 将LLM生成的token序列转换为高质量的梅尔频谱

### Frontend
- **文件位置**: [`cosyvoice/cli/frontend.py`](cosyvoice/cli/frontend.py)
- **功能**: 文本和语音的预处理，包括文本归一化、音素转换、语音token提取和说话人嵌入提取
- **特点**: 支持中英文混合文本处理

### 强化学习模块
- **文件位置**: [`grpo/`](grpo/)目录
- **核心文件**:
  - [`grpo_utils.py`](grpo/grpo_utils.py): GRPO算法实现和批处理推理
  - [`reward_func.py`](grpo/reward_func.py): 多奖励函数实现
  - [`reward_server.py`](grpo/reward_server.py): 分布式奖励服务器
- **功能**: 通过多奖励强化学习优化TTS系统的情感表达能力

## 实验结果

在 `seed-tts-eval中文测试集` 上进行评估，为与原版评估保持一致，未使用 `--phoneme` 参数进行推理。
**CER**: 字符错误率 (越低越好 $\downarrow$) | **SIM**: 相似度 (越高越好 $\uparrow$)

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

## 项目结构

```
GLM-TTS/
├── glmtts_inference.py              # 主推理脚本，包含完整的推理流程
├── glmtts_inference.sh              # 预训练模型推理脚本
├── configs/                         # 配置文件目录
│   ├── spk_prompt_dict.yaml         # 说话人提示字典
│   ├── lora_adapter_configV3.1.json # LoRA适配器配置
│   ├── G2P_able_1word.json          # 单字音素转换配置
│   ├── G2P_all_phonemes.json        # 全音素列表
│   ├── G2P_replace_dict.jsonl       # 音素替换字典
│   └── custom_replace.jsonl         # 自定义替换规则
├── cosyvoice/                       # Cosyvoice模块
│   ├── cli/
│   │   └── frontend.py              # 文本和语音前端处理
│   └── utils/                       # 工具函数
├── examples/                        # 示例数据
│   ├── *.jsonl                      # 示例jsonl文件
│   └── prompt/                      # 提示音频目录
│       ├── *.wav                    # 提示音频（仅限科研使用）
│       └── LICENSE                  # 音频文件许可证
├── flow/                            # Flow模型相关
│   ├── dit.py                       # Diffusion Transformer实现
│   ├── flow.py                      # 流式Flow模型
│   └── modules.py                   # Flow模型基础模块
├── grpo/                            # 强化学习模块
│   ├── grpo_utils.py                # GRPO算法实现
│   ├── reward_func.py               # 多奖励函数
│   ├── reward_server.py             # 分布式奖励服务器
│   ├── train_ds_grpo.py             # GRPO训练脚本
│   └── data/                        # 训练数据和配置
├── llm/                             # 大语言模型相关
│   └── glmtts.py                    # GLM-TTS LLM实现
├── frontend/                        # 前端模型文件
│   ├── campplus.onnx                # 说话人嵌入模型
│   └── cosyvoice_frontend.yaml      # 前端配置
├── tools/                           # 工具脚本
│   ├── gradio_app.py                # Gradio交互界面
│   ├── ffmpeg_speech_control.py     # 音频处理工具
│   └── flow_reconstruct.py          # 音频重建
└── utils/                           # 通用工具
    ├── tts_model_util.py            # TTS模型工具
    ├── yaml_util.py                 # YAML配置加载工具
    ├── audio.py                     # 音频处理工具
    ├── seed_util.py                 # 随机种子工具
    ├── block_mask_util.py           # 块掩码工具
    ├── vocos_util.py                # Vocos声码器工具
    ├── hift_util.py                 # Hift声码器工具
    ├── whisper_models/              # Whisper模型相关
    └── glm_g2p.py                # 文字到音素转换
```

## 致谢

我们感谢以下开源项目的支持：

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) - 提供前端处理框架和高质量声码器
- [Llama](https://github.com/meta-llama/llama) - 提供基础语言模型架构
- [Vocos](https://github.com/charactr-platform/vocos) - 提供高质量声码器
- [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) - 强化学习算法实现灵感


## 引用

如果您觉得 GLM-TTS 对您的研究有用，请引用我们的技术报告：

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