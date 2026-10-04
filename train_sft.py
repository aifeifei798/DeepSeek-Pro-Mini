import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (PretrainedConfig, PreTrainedModel, GenerationMixin,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast

# =========================================================
# 硬件与底层性能优化开关
# =========================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_BIN_PATH = "dataset/sft_inputs.bin"
LABEL_BIN_PATH = "dataset/sft_labels.bin"
PRETRAIN_MODEL_DIR = "./my_deepseek_pretrain_model"
SFT_SAVE_DIR = "./my_deepseek_sft_model"
CHECKPOINT_DIR = "./checkpoints_sft/latest"
SAVE_EVERY = 500

tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_DIR,
                                          trust_remote_code=True)
VOCAB_SIZE = max(len(tokenizer), getattr(tokenizer, "vocab_size", 0))


# =========================================================
# 架构定义（保持与预训练基座 100% 相同）
# =========================================================
class MiniDeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_pro_mini"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = VOCAB_SIZE
        self.hidden_size = 512
        self.num_layers = 8
        self.num_heads = 8
        self.head_dim = 64
        self.q_lora_rank = 64
        self.kv_lora_rank = 64
        self.o_lora_rank = 64
        self.n_routed_experts = 8
        self.num_experts_per_tok = 2
        self.n_shared_experts = 1
        self.moe_intermediate_size = 768
        self.swiglu_limit = 10.0
        self.routed_scaling_factor = 2.5
        self.tie_word_embeddings = True


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
        self.gate = nn.Linear(config.hidden_size,
                              config.n_routed_experts,
                              bias=False)
        self.experts = nn.ModuleList([
            BoundedSwiGLU(config.hidden_size, config.moe_intermediate_size,
                          config.swiglu_limit)
            for _ in range(config.n_routed_experts)
        ])
        self.shared_expert = BoundedSwiGLU(
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            config.swiglu_limit)

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.config.hidden_size)
        scores = torch.sqrt(F.softplus(self.gate(x_flat)) + 1e-6)
        weights, indices = torch.topk(scores,
                                      self.config.num_experts_per_tok,
                                      dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.config.num_experts_per_tok):
            for e_idx in range(self.config.n_routed_experts):
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    routed_out[mask] += self.experts[e_idx](
                        x_flat[mask]) * weights[mask, k].unsqueeze(-1)

        out = (routed_out *
               self.config.routed_scaling_factor) + self.shared_expert(x_flat)
        return out.view(orig_shape)


class MiniMLA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_down = nn.Linear(config.hidden_size,
                                config.q_lora_rank,
                                bias=False)
        self.q_up = nn.Linear(config.q_lora_rank,
                              config.num_heads * config.head_dim,
                              bias=False)
        self.kv_down = nn.Linear(config.hidden_size,
                                 config.kv_lora_rank,
                                 bias=False)
        self.k_up = nn.Linear(config.kv_lora_rank,
                              config.num_heads * config.head_dim,
                              bias=False)
        self.v_up = nn.Linear(config.kv_lora_rank,
                              config.num_heads * config.head_dim,
                              bias=False)
        self.o_down = nn.Linear(config.num_heads * config.head_dim,
                                config.o_lora_rank,
                                bias=False)
        self.o_up = nn.Linear(config.o_lora_rank,
                              config.hidden_size,
                              bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_up(self.q_down(x)).view(b, s, self.num_heads,
                                           self.head_dim).transpose(1, 2)
        kv = self.kv_down(x)
        k = self.k_up(kv).view(b, s, self.num_heads,
                               self.head_dim).transpose(1, 2)
        v = self.v_up(kv).view(b, s, self.num_heads,
                               self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
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
    _tied_weights_keys = {"lm_head.weight": "embed.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekProBlock(config) for _ in range(config.num_layers)])
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm_f(h))
        loss = None
        if labels is not None:
            # 忽略所有 -100 掩码
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # 自回归错开一位计算 Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size),
                           shift_labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


AutoModelForCausalLM.register(MiniDeepSeekConfig, ToyDeepSeekProForCausalLM)


# =========================================================
# SFT 二进制内存映射读取器
# =========================================================
class MemmapSFTDataset(Dataset):

    def __init__(self, input_bin, label_bin, seq_len=512):
        self.seq_len = seq_len
        self.inputs = np.memmap(input_bin, dtype=np.uint32, mode="r")
        self.labels = np.memmap(label_bin, dtype=np.int32, mode="r")
        self.num_samples = len(self.inputs) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.from_numpy(self.inputs[start:end].astype(np.int64))
        y = torch.from_numpy(self.labels[start:end].astype(np.int64))
        return x, y


# =========================================================
# 检查点管理
# =========================================================
def save_checkpoint(model, optimizer, scheduler, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    raw_model.tie_weights()
    try:
        raw_model.save_pretrained(save_dir, safe_serialization=True)
    except Exception:
        raw_model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)
    state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(state, os.path.join(save_dir, "trainer_state.pt"))
    print(f"\n💾 [CheckPoint] 已在 Step {step} 成功固化 SFT 快照至: {save_dir}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    state_file = os.path.join(checkpoint_dir, "trainer_state.pt")
    if not os.path.exists(state_file):
        return 0
    print(f"🔄 发现 SFT 历史检查点，正在从 {checkpoint_dir} 恢复现场...")
    raw_model = getattr(model, "_orig_mod", model)
    loaded_model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    raw_model.load_state_dict(loaded_model.state_dict())
    del loaded_model

    state = torch.load(state_file, map_location=device)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    step = state["step"]
    print(f"✅ SFT 现场恢复成功！直接从 Step {step + 1} 接着跑！\n")
    return step


# SFT 对话能力实测抽检
def preview_chat(model, tokenizer, question="中国的首都是哪里？"):
    model.eval()
    prompt = f"<｜User｜>{question}<｜Assistant｜>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(inputs["input_ids"],
                             max_new_tokens=40,
                             use_cache=False,
                             temperature=0.3,
                             do_sample=True,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id)
    model.train()
    res = tokenizer.decode(out[0], skip_special_tokens=True)
    return res[len(prompt):]


# =========================================================
# SFT 训练主引擎
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_BIN_PATH) or not os.path.exists(
            LABEL_BIN_PATH):
        print("❌ 找不到 SFT 二进制文件，请先运行 python preprocess_sft.py！")
        sys.exit(1)

    print(f"📦 正在载入已训练 2.45 亿 Token 的 Base 基座: {PRETRAIN_MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL_DIR).to(device)
    print("✅ Base 基座载入完毕！准备进行灵魂对齐微调。")

    # 🚀 SFT 参数配置（温柔退火，保护知识）
    SEQ_LEN = 512
    BATCH_SIZE = 20  # 显存安全甜点位
    GRAD_ACCUM_STEPS = 4  # 每步 40,960 tokens
    TOTAL_STEPS = 3500  # 4000 步约吃 32 万条对话，耗时约 25 分钟

    # SFT 专用温和学习率 (2e-4 -> 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=2e-4,
                                  weight_decay=1e-2,
                                  fused=(device == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-5)

    start_step = load_checkpoint(model, optimizer, scheduler, CHECKPOINT_DIR)

    try:
        print("🔥 激活 torch.compile Triton 算子融合编译...")
        model = torch.compile(model)
    except Exception as e:
        print(f"⚠️ torch.compile 跳过: {e}")

    dataset = MemmapSFTDataset(INPUT_BIN_PATH, LABEL_BIN_PATH, seq_len=SEQ_LEN)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)
    amp_dtype = torch.bfloat16 if (
        device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n🚀 【SFT 对话微调引擎】全速启动！")
    print(
        f"📊 总对话样本数: {len(dataset)} 条，单步吞吐: {BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN} tokens"
    )
    print(
        f"⚡ 当前步数: {start_step} / {TOTAL_STEPS} | 每 {SAVE_EVERY} 步自动存盘并抽测对话能力\n"
    )

    step = start_step
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

                bx, by = bx.to(device,
                               non_blocking=True), by.to(device,
                                                         non_blocking=True)

                with torch.amp.autocast(device_type=device,
                                        dtype=amp_dtype,
                                        enabled=(device == "cuda")):
                    outputs = model(input_ids=bx, labels=by)
                    loss = outputs.loss / GRAD_ACCUM_STEPS

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            running_loss += accum_loss

            if step % 50 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (50 * BATCH_SIZE * GRAD_ACCUM_STEPS *
                                  SEQ_LEN) / elapsed
                avg_loss = running_loss / 50
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Step [{step:05d}/{TOTAL_STEPS}] | Loss: {avg_loss:.4f} | LR: {lr:.6f} | ⚡ 吞吐: \033[1;32m{tokens_per_sec:.0f} tokens/s\033[0m"
                )
                running_loss = 0.0
                start_time = time.time()

            # 每 500 步存盘并现场面试它一次
            if step % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler, step,
                                CHECKPOINT_DIR)
                print("\n" + "=" * 50)
                print(f"💬 [Step {step} 对话能力现场抽测]")
                test_q = "中国的首都是哪里？"
                raw_model = getattr(model, "_orig_mod", model)
                print(f"  问: {test_q}")
                print(
                    f"  答: \033[1;32m{preview_chat(raw_model, tokenizer, test_q)}\033[0m"
                )
                print("=" * 50 + "\n")
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n🛑 检测到手动中断信号，正在为您保存 SFT 现场...")
        save_checkpoint(model, optimizer, scheduler, step, CHECKPOINT_DIR)

    if step >= TOTAL_STEPS:
        save_checkpoint(model, optimizer, scheduler, step, SFT_SAVE_DIR)
        print(f"\n🎉 恭喜！SFT 对话指令微调圆满完成！最终对话模型已保存在: {SFT_SAVE_DIR}")
