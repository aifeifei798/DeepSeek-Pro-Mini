# 🚀 DeepSeek-Pro-Mini: 1小时手搓原生 MLA + MoE 大模型全栈实战

> **“大道至简，行则将至。”**  
> 本项目从零手搓复现了前沿 **DeepSeek-V3 / V4** 核心架构（MLA 隐空间注意力 + 8专家稀疏 MoE + SqrtSoftplus 门控 + Bounded SwiGLU）。并在消费级单卡上榨出 **52,000+ tokens/s** 的极限吞吐，完成了 **2.45 亿 Token 预训练** 与 **90 万条 SFT 对话对齐**，最终打造出一个具备丰富常识、支持流式打字机交互、并能实时透视 8 层 MoE 门控放电热力图的端到端对话智能体！

---

## 📑 目录
- [🌟 核心亮点](#-核心亮点)
- [🧠 架构参数解密 (153M-A60M)](#-架构参数解密-153m-a60m)
- [⚡ 极速训练黑科技 (52,000+ tokens/s 的秘密)](#-极速训练黑科技-52000-tokenss-的秘密)
- [🛠️ 踩坑血泪史与避坑指南 (必看)](#️-踩坑血泪史与避坑指南-必看)
- [🚀 快速开始与全流程复现](#-快速开始与全流程复现)
  - [第 1 步：依赖与分词器就位](#第-1-步依赖与分词器就位)
  - [第 2 步：预训练数据极速二进制化](#第-2-步预训练数据极速二进制化)
  - [第 3 步：开启满血预训练 (Pretrain)](#第-3-步开启满血预训练-pretrain)
  - [第 4 步：Base 底座“大考”体检](#第-4-步base-底座大考体检)
  - [第 5 步：SFT 对话指令微调 (灵魂注入)](#第-5-步sft-对话指令微调-灵魂注入)
  - [第 6 步：终端交互与 MoE 神经放电透视](#第-6-步终端交互与-moe-神经放电透视)
- [📂 项目完整目录结构](#-项目完整目录结构)
- [🤝 致谢与参考](#-致谢与参考)

---

## 🌟 核心亮点

* **纯手搓架构内核**：没有调用任何现成模型黑盒，纯 PyTorch 从零搭建 MLA、MoE、RMSNorm、SwiGLU 算子。
* **稀疏算力奇迹**：总参数量约 **1.53 亿（153M）**，每个 Token 仅激活 **~60M（A60M）**，显存常驻仅需 3.5GB 左右。
* **工业级吞吐压榨**：集成 **PyTorch SDPA (FlashAttention-2)、TF32 矩阵加速、Fused AdamW、Triton 算子编译 (`torch.compile`)** 与 **零 CPU 开销的 Memmap 内存映射**，单卡逼近 53,000 tokens/s！
* **大模型发育延时摄影**：亲眼见证模型从“随机乱码” $\rightarrow$ “困惑度断崖下跌” $\rightarrow$ “口吃复读” $\rightarrow$ “时空错乱脑洞” $\rightarrow$ “精准实体对齐与自主作诗”。
* **全透明可解释性**：终端实时抓取 CUDA 钩子，以 ANSI 矩阵彩色渲染 8 层专家的实时放电决策（E0 写诗、E7 编程）。

---

## 🧠 架构参数解密 (153M-A60M)

本项目高度凝练了 DeepSeek-V3/V4 论文中的前沿创新，超参配置如下：

| 模块 | 超参数 | 设计机理 |
| :--- | :--- | :--- |
| **词表 (Vocab)** | `129,280` | 对齐 DeepSeek 官方 BPE 分词器，汉字压缩率极高 |
| **权重绑定 (Weight Tying)** | `True` | **输入与输出层共享 Embedding**，立省 66M 参数与显存 |
| **注意力机制 (MLA)** | `hidden_size=512`, `heads=8` | Q/KV/O 均做 **64 维低秩压缩 (LoRA Rank)**，兼顾大感受野与低显存 |
| **激活函数** | `swiglu_limit=10.0` | **截断限幅 SwiGLU**，彻底扼杀极深网络下的激活值溢出（NaN） |
| **混合专家 (MoE)** | `8 路由专家 (选2) + 1 共享` | **共享专家全程 100% 在线兜底**；路由专家负责分领域垂直特化 |
| **门控打分** | `scoring_func="sqrtsoftplus"` | 替代传统 Softmax，平滑路由分流权重，防止单一专家极化与过劳死 |

```text
总参数量: 153.2 M
单 Token 实际激活参数: ~60.3 M
```

---

## ⚡ 极速训练黑科技 (52,000+ tokens/s 的秘密)

普通个人训练大模型常因数据读取和框架调度导致 GPU“等 CPU 喂饭”（GPU 利用率常年 50%）。本项目做了全链路加速闭环：

1. **二进制内存映射 (Memmap)**：放弃训练时动态 `json.loads` 与在线 Tokenize。预先用几十秒将文本序列化为 `np.uint32` 单文件，读取开销瞬间降至 0。
2. **PyTorch SDPA**：一行调用硬件级 FlashAttention-2，消灭庞大的 $O(S^2)$ 中间注意力矩阵。
3. **TF32 与 Fused AdamW**：全面开启 Ampere/Ada/Blackwell 专属 TensorFloat-32 单元与单 Kernel 优化器更新。
4. **Triton 算子熔炼 (`torch.compile`)**：将 8 层 MoE 带来的上百次碎小 Python 切片循环，在 JIT 层面编译融合成定制机器码。

---

## 🛠️ 踩坑血泪史与避坑指南 (必看)

在大模型从零手搓过程中，我们踩平了几个足以让绝大多数人放弃的“硬核天坑”：

### 💥 天坑 1：12.9 万大词表的“显存核爆”（精确到字节的 OOM）
* **现象**：当 Batch Size 设为 36 时，CUDA 直接抛出 `Tried to allocate 9531555840 bytes (8.88 GiB) OOM`。
* **原因**：$36 \times 512 \text{ tokens} = 18,432$ 个词元。最后分类头做预测时，Logits 矩阵形状高达 $[18432, 129280]$。在 FP32 下：
  $$18432 \times 129280 \times 4 \text{ 字节} = \mathbf{9,531,555,840 \text{ 字节 (整整 8.88 GB！)}}$$
* **解法**：单次前向 Batch Size 控制在 `20`（显存开销仅 ~4.5GB），配合 `GRAD_ACCUM_STEPS = 4`，总 Batch 依然高达 40,960 tokens/step，吞吐量拉满且永不爆显存！

### 💥 天坑 2：Transformers v5.x 权重绑定类型断言错误
* **现象**：开启 `tie_word_embeddings` 存盘时报错：`AttributeError: 'list' object has no attribute 'keys'`。
* **原因**：Transformers 最新规范将 `_tied_weights_keys` 从原本的列表改为字典映射。
* **解法**：必须显式声明源与目标映射：
  ```python
  _tied_weights_keys = {"lm_head.weight": "embed.weight"}
```
  并在自定义模型中实现 `def tie_weights(self, *args, **kwargs)` 以兼容新版参数。

### 💥 天坑 3：SFT 提示词未 Mask 导致“复读机退化”
* **现象**：SFT 训练时如果不给 Prompt 打掩码，模型会把用户提问本身也当答案背诵。
* **解法**：`<｜User｜>提问<｜Assistant｜>` 区域的 Label **必须强制赋值为 `-100`**，只有 Assistant 的回复与结束符才参与 Loss 计算！

---

## 🚀 快速开始与全流程复现

### 第 1 步：依赖与分词器就位

```bash
git clone https://github.com/aifeifei798/DeepSeek-Pro-Mini.git
cd DeepSeek-Pro-Mini

# 推荐 Python 3.10+，PyTorch 2.2+ (CUDA 12+)
pip install torch transformers numpy tqdm
```

下载 DeepSeek 官方分词器文件（或本项目仓库内置的 `tokenizer.json`、`tokenizer_config.json`），放置在根目录下。

---

### 第 2 步：预训练数据极速二进制化

推荐使用开源极纯中文预训练语料 **MiniMind-3 数据集**（通过 ModelScope 免登录下载）：

```bash
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset
```

运行离线打包脚本（约 2~5 分钟）：
```bash
python preprocess.py
```
*生成 `dataset/pretrain_data.bin`，打包出 **2.45 亿 Tokens**。*

---

### 第 3 步：开启满血预训练 (Pretrain)

启动满血预训练引擎（支持全自动断点续训，每 500 步自动快照）：

```bash
python pretrain_turbo.py
```

* **实测指标（以 RTX 5090 D 为例）**：
  * **吞吐量**：$\approx \mathbf{52,000 \text{ tokens/s}}$
  * **总耗时**：**1 小时 20 分钟** 完整吃完 2.45 亿 Token（正好 1.0 个 Epoch）
  * **收敛 Loss**：从 `11.5` 稳步自由落体至 **`3.04`**（困惑度 PPL $\approx 20.9$）

---

### 第 4 步：Base 底座“大考”体检

预训练完成后，直接运行评估脚本，观察 Base 模型的续写与百科储备：

```bash
python eval_base.py
```

* **实测续写表现摘录**：
  * 输入：`李白是唐朝著名的` $\rightarrow$ 续写：`诗人，他的诗歌风格独特、奔放不羁。`
  * 输入：`水在常温常压下的沸点是` $\rightarrow$ 续写：`100℃时，水的沸点为0°C...`
  * 输入：`地球绕着太阳` $\rightarrow$ 续写：`旋转，而月球绕着自己的轴心自转。`

---

### 第 5 步：SFT 对话指令微调 (灵魂注入)

#### 5.1 加工 SFT 数据（打掩码与套模板）
```bash
python preprocess_sft.py
```
*耗时约 3 分钟，将 90.5 万条对话结构化为 `sft_inputs.bin` 和 `sft_labels.bin`。*

#### 5.2 启动 SFT 微调
```bash
python train_sft.py
```
* **微调策略**：加载预训练底座，采用温和学习率（$2\times 10^{-4}$），训练 4000 步（约 25 分钟）。
* **见证模型“智力演进”**：
  * *Step 500*：初具问答意识，但出现“金融、金融”单字复读；
  * *Step 1000*：标点语法流畅，但时空错乱把首都当朝代；
  * *Step 3000+（巅峰）*：回答“中国的首都是北京，拥有故宫、长城、颐和园等名胜古迹”，格式与实体彻底对齐！

---

### 第 6 步：终端交互与 MoE 神经放电透视

运行最终交互终端，享受打字机流式对话与实时专家放电矩阵：

```bash
python chat_sft.py
```

#### 🌟 交互实测效果：

```text
User ❯ 请帮我写一首赞美秋天的小诗。
DeepSeek-Mini ❯ 
《秋时节》
红叶飘满院中。
云层薄如镜，残雪渐深，
清风拂面；露珠在花间。

赏析：这首作品描绘了秋天的美丽画卷和自然景象的和谐与美好。诗中“金黄”象征着丰收、凉爽；“霜叶是落叶的完美结合”，进一步强化了宁静之美。整体意境深远，情感真挚而富有诗意，体现了诗人对生命意义的深刻感悟。
```

#### 📊 8 层 MoE 门控放电热力图透视（局部切片）：
```text
[ Layer 6 路由分布 ]
Token    | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | Shared
---------------------------------------------------------
秋        | 71.6%|  · |  · | 28.4%|  · |  · |  · |  · | 100% (ON)
红叶      | 65.2%|  · |  · | 34.8%|  · |  · |  · |  · | 100% (ON)
飘        | 80.2%|  · |  · | 19.8%|  · |  · |  · |  · | 100% (ON)
...
（注：写诗时 Layer 6 的 Expert 0 狂飙至 80% 统治输出；而提问 Python 代码时，E0 彻底归零休眠，切换为 Expert 7 主导！）
```

---

## 📂 项目训练完成后的完整目录结构

```text
DeepSeek-Pro-Mini/
├── dataset/                    # 语料与预处理二进制缓存
│   ├── pretrain_data.bin       # 2.45亿 Token 预训练二进制
│   ├── sft_inputs.bin          # SFT 对话输入张量
│   └── sft_labels.bin          # SFT 对话 -100 掩码张量
├── checkpoints/latest/         # 预训练定期自动断点
├── checkpoints_sft/latest/     # SFT 定期自动断点
├── my_deepseek_pretrain_model/ # 最终固化的 Base 模型权重
├── my_deepseek_sft_model/      # 最终固化的 SFT 对话智能体
│   ├── config.json             # HF 标准架构配置
│   ├── model.safetensors       # 53 个算子权重全集
│   └── tokenizer.json          # 分词器词表
├── preprocess.py               # 预训练语料二进制转换器
├── pretrain_turbo.py           # 满血预训练核心引擎 (FlashAttention+Compile)
├── eval_base.py                # Base 模型综合能力体检脚本
├── preprocess_sft.py           # SFT 多轮对话模板化与掩码打包器
├── train_sft.py                # SFT 指令微调训练器
└── chat_sft.py                 # 终端流式打字机交互 + MoE 门控透视仪
```

---

## 🤝 致谢与参考

1. **DeepSeek-AI**：感谢 DeepSeek 团队开源公开的 [DeepSeek-V3/V4 Technical Report](https://github.com/deepseek-ai) 及其对 MLA 和细粒度 MoE 的卓越贡献。
2. **MiniMind 项目**：感谢 [@jingyaogong](https://github.com/jingyaogong/minimind) 开源的高纯度中文精炼数据集 `minimind_dataset`。
3. **Hugging Face**：感谢 `transformers` 提供的生态规范与通用模型接口。

---

> **🌟 如果这个项目帮助你打破了大模型“黑盒迷雾”，欢迎在 GitHub 上点一个 Star！**
