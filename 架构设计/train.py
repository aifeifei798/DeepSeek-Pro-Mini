import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from data import raw_data

# ==========================================
# 1. 硬件自适应与高效配置
# ==========================================
device = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"🚀 使用计算设备: {device}")

tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

VOCAB_SIZE = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0))

# ==========================================
# 2. 准确度质跃点：SFT Prompt Masking 显存常驻构造
# ==========================================
def build_fast_gpu_dataset(texts, tokenizer, max_length=48):
    all_inputs, all_labels = [], []
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0

    for text in texts:
        # 分割 Question 和 Answer
        q_text, a_text = text.split(" 答：")
        prompt_text = q_text + " 答："

        # 编码 Prompt 与整体
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(text) + [eos_id]

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        
        pad_len = max_length - len(full_ids)
        input_ids = full_ids + [pad_id] * pad_len
        
        # 核心优化：Prompt 区域的 target 赋为 -100 (不学问题，只学答案！)
        prompt_len = min(len(prompt_ids), max_length)
        target_ids = full_ids[1:] + [pad_id] + [-100] * pad_len
        target_ids = target_ids[:max_length]
        
        # 将 Prompt 对应位置全部置为 -100
        for i in range(prompt_len - 1):
            target_ids[i] = -100

        all_inputs.append(input_ids[:-1])
        all_labels.append(target_ids[:-1])

    # 整个数据集直接打成 CUDA Tensor 驻留在显存！
    gpu_inputs = torch.tensor(all_inputs, dtype=torch.long, device=device)
    gpu_labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    return gpu_inputs, gpu_labels

gpu_inputs, gpu_labels = build_fast_gpu_dataset(raw_data, tokenizer, max_length=48)
print(f"📦 数据已构建完成并预加载至显存，样本数: {gpu_inputs.shape[0]}")

# ==========================================
# 3. 架构定义 (保持标准不变)
# ==========================================
class MiniDeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_pro_mini"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        q_lora_rank=32,
        kv_lora_rank=32,
        o_lora_rank=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        swiglu_limit=10.0,
        routed_scaling_factor=2.5,
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

AutoConfig.register("deepseek_pro_mini", MiniDeepSeekConfig)

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
# 4. 极速融合训练器 (AMP + Fused AdamW + Cosine LR)
# ==========================================
if __name__ == "__main__":
    SAVE_DIR = "./my_deepseek_pro_model"
    
    config = MiniDeepSeekConfig(vocab_size=VOCAB_SIZE)
    model = ToyDeepSeekProForCausalLM(config).to(device)

    # 关键速度优化：Fused AdamW (如果支持)
    use_fused = (device == "cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=4e-3, 
        weight_decay=1e-2, 
        fused=use_fused
    )

    epochs = 80
    batch_size = 25  # 100个样本放大到25，一轮仅需 4 次 Step，速度翻倍！
    num_samples = gpu_inputs.shape[0]

    # 余弦退火调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    # AMP 半精度类型判断 (支持 BF16 则用 BF16，否则 FP16)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    print(f"\n--- 1. 开始极速训练 (AMP: {amp_dtype}, Fused: {use_fused}) ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # 显存内打乱索引
        perm = torch.randperm(num_samples, device=device)
        
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx, by = gpu_inputs[idx], gpu_labels[idx]

            optimizer.zero_grad(set_to_none=True)

            # AMP 混合精度上下文
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                outputs = model(input_ids=bx, labels=by)
                loss = outputs.loss

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / (num_samples // batch_size)
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {avg_loss:.4f} - LR: {curr_lr:.5f}")

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0
    print(f"⚡ 训练完毕！纯训练耗时: {elapsed_time:.2f} 秒！")

    print("\n--- 2. 固化保存 ---")
    model.eval()
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ 模型与分词器已固化到: {SAVE_DIR}")

    print("\n--- 3. 验证精度（测试太阳方向与数学题）---")
    del model
    torch.cuda.empty_cache()

    loaded_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    loaded_model = AutoModelForCausalLM.from_pretrained(SAVE_DIR).to(device)

    test_cases = [
        "问：太阳从哪个方向升起？ 答：",
        "问：太阳从哪个方向落下？ 答：",  # 对比测试：看还会不会混淆！
        "问：一加一等于几？ 答：",
        "问：中国的首都是哪里？ 答："
    ]
    
    for prompt in test_cases:
        inputs = loaded_tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = loaded_model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            use_cache=False,
            eos_token_id=loaded_tokenizer.eos_token_id,
            pad_token_id=loaded_tokenizer.pad_token_id
        )
        print("输入:", prompt)
        print("输出:", loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True))
        print("-" * 40)