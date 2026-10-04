
import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    PretrainedConfig, 
    PreTrainedModel, 
    GenerationMixin, 
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# ==========================================
# 1. 硬件自适应与路径检测
# ==========================================
device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"🚀 训练设备: {device}")

# 兼容检测 dataset 或 data 目录
DATA_PATH = (
    "dataset/pretrain_t2t_mini.jsonl" 
    if os.path.exists("dataset/pretrain_t2t_mini.jsonl") 
    else "data/pretrain_t2t_mini.jsonl"
)
SAVE_DIR = "./my_deepseek_pretrain_model"
print(f"📂 锁定预训练语料: {DATA_PATH}")

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
VOCAB_SIZE = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0))

# ==========================================
# 2. 实战级 160M-A60M DeepSeek 架构配置
# ==========================================
class MiniDeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_pro_mini"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=512,           # 提升到 512
        num_layers=8,              # 8层深度
        num_heads=8,
        head_dim=64,
        q_lora_rank=64,
        kv_lora_rank=64,
        o_lora_rank=64,
        n_routed_experts=8,        # 8个专家
        num_experts_per_tok=2,     # 选2个
        n_shared_experts=1,        # 1个共享
        moe_intermediate_size=768, # 专家容量加宽
        swiglu_limit=10.0,
        routed_scaling_factor=2.5,
        tie_word_embeddings=True,  # 权重绑定，立省66M显存！
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.o_lora_rank = o_lora_rank
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.swiglu_limit = swiglu_limit
        self.routed_scaling_factor = routed_scaling_factor
        self.tie_word_embeddings = tie_word_embeddings

AutoConfig.register("deepseek_pro_mini", MiniDeepSeekConfig)

# 基础算子实现
class BoundedSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, limit=10.0):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)
        self.limit = limit

    def forward(self, x):
        gate = torch.clamp(F.silu(self.w_gate(x)), max=self.limit)
        up = torch.clamp(self.w_up(x), min=-self.limit, max=self.limit)
        return self.w_down(gate * up)

class DeepSeekMiniMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([
            BoundedSwiGLU(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
            for _ in range(config.n_routed_experts)
        ])
        self.shared_expert = BoundedSwiGLU(
            config.hidden_size, 
            config.moe_intermediate_size * config.n_shared_experts, 
            config.swiglu_limit
        )

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.config.hidden_size)

        scores = torch.sqrt(F.softplus(self.gate(x_flat)) + 1e-6)
        weights, indices = torch.topk(scores, self.config.num_experts_per_tok, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.config.num_experts_per_tok):
            for e_idx in range(self.config.n_routed_experts):
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    routed_out[mask] += self.experts[e_idx](x_flat[mask]) * weights[mask, k].unsqueeze(-1)

        out = (routed_out * self.config.routed_scaling_factor) + self.shared_expert(x_flat)
        return out.view(orig_shape)

class MiniMLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_down = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_up = nn.Linear(config.q_lora_rank, config.num_heads * config.head_dim, bias=False)
        self.kv_down = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.k_up = nn.Linear(config.kv_lora_rank, config.num_heads * config.head_dim, bias=False)
        self.v_up = nn.Linear(config.kv_lora_rank, config.num_heads * config.head_dim, bias=False)
        self.o_down = nn.Linear(config.num_heads * config.head_dim, config.o_lora_rank, bias=False)
        self.o_up = nn.Linear(config.o_lora_rank, config.hidden_size, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_up(self.q_down(x)).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_down(x)
        k = self.k_up(kv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(kv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.full((s, s), float('-inf'), device=x.device), diagonal=1)
        attn = F.softmax(scores + causal_mask, dim=-1)

        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_up(self.o_down(attn_out))

class DeepSeekProBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = MiniMLA(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.moe = DeepSeekMiniMoE(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class ToyDeepSeekProForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniDeepSeekConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekProBlock(config) for _ in range(config.num_layers)])
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 重点：权重绑定 (Weight Tying)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed.weight

        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm_f(h))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

AutoModelForCausalLM.register(MiniDeepSeekConfig, ToyDeepSeekProForCausalLM)


# ==========================================
# 3. 工业级流式打包读取器 (Constant Length Packing)
# ==========================================
class StreamingPretrainDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, seq_len=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        token_buffer = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue
                    # 编码单行并追加结束符
                    ids = self.tokenizer.encode(text) + [self.eos_id]
                    token_buffer.extend(ids)

                    # 当蓄水池满 512+1 个 token，就切出一个样本
                    while len(token_buffer) >= (self.seq_len + 1):
                        chunk = token_buffer[:self.seq_len + 1]
                        token_buffer = token_buffer[self.seq_len:]
                        
                        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                        labels = torch.tensor(chunk[1:], dtype=torch.long)
                        yield input_ids, labels
                except Exception:
                    continue

# 验证测试文本续写
def preview_generate(model, tokenizer, prompt="从前有一座山，", max_new_tokens=25):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=True,
            top_p=0.85,
            temperature=0.8
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ==========================================
# 4. 预训练主引擎
# ==========================================
if __name__ == "__main__":
    cfg = MiniDeepSeekConfig(vocab_size=VOCAB_SIZE)
    model = ToyDeepSeekProForCausalLM(cfg).to(device)

    # 计算真实参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型初始化完成！总参数量: {total_params / 1e6:.2f} M (百万)")

    # 批大小与上下文配置
    SEQ_LEN = 512
    BATCH_SIZE = 12       # 显存富余可调至 16
    GRAD_ACCUM_STEPS = 2  # 累积后等效 Batch Size = 24
    TOTAL_STEPS = 6000    # 可以随时 Ctrl+C，模型会自动保存

    dataset = StreamingPretrainDataset(DATA_PATH, tokenizer, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=8e-4, 
        weight_decay=1e-2, 
        fused=(device == "cuda")
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n🚀 开始真实语料从零预训练 (总步数: {TOTAL_STEPS}, 上下文: {SEQ_LEN})...")
    print("💡 提示：随时可以按 Ctrl+C，程序会自动保存当前进度！\n")

    step = 0
    running_loss = 0.0
    start_time = time.time()
    data_iter = iter(dataloader)

    try:
        model.train()
        while step < TOTAL_STEPS:
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for _ in range(GRAD_ACCUM_STEPS):
                try:
                    bx, by = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    bx, by = next(data_iter)

                bx, by = bx.to(device), by.to(device)

                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                    outputs = model(input_ids=bx, labels=by)
                    loss = outputs.loss / GRAD_ACCUM_STEPS

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += accum_loss

            # 每 50 步打印一次吞吐与 Loss
            if step % 50 == 0:
                tokens_per_sec = (50 * BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN) / (time.time() - start_time)
                avg_loss = running_loss / 50
                lr = scheduler.get_last_lr()[0]
                print(f"Step [{step:05d}/{TOTAL_STEPS}] | Loss: {avg_loss:.4f} | LR: {lr:.6f} | 吞吐: {tokens_per_sec:.0f} tokens/s")
                running_loss = 0.0
                start_time = time.time()

            # 每 500 步试写一段话，看看从“乱码”到“中文”的进化过程
            if step % 500 == 0:
                print("\n" + "=" * 50)
                print(f"🎨 [Step {step} 文本生成抽检]")
                print(preview_generate(model, tokenizer, "人工智能是"))
                print("=" * 50 + "\n")
                model.train()

    except KeyboardInterrupt:
        print("\n🛑 检测到中断信号，准备安全保存当前权重...")

    # 保存预训练好的底座模型
    print(f"\n💾 正在保存预训练底座到: {SAVE_DIR}")
    model.eval()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"🎉 预训练底座已就绪！")