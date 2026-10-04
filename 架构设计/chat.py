import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (PretrainedConfig, PreTrainedModel, GenerationMixin,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast

MODEL_DIR = "./my_deepseek_pro_model"

device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")


# =========================================================
# 1. 架构定义与 Hugging Face 注册（独立自包含）
# =========================================================
class MiniDeepSeekConfig(PretrainedConfig):
    model_type = "deepseek_pro_mini"

    def __init__(self,
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
                 **kwargs):
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal_mask = torch.triu(torch.full((s, s),
                                            float('-inf'),
                                            device=x.device),
                                 diagonal=1)
        attn = F.softmax(scores + causal_mask, dim=-1)

        attn_out = torch.matmul(attn,
                                v).transpose(1, 2).contiguous().view(b, s, -1)
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
        self.layers = nn.ModuleList(
            [DeepSeekProBlock(config) for _ in range(config.num_layers)])
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(self.norm_f(h))
        return CausalLMOutputWithPast(loss=None, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


AutoModelForCausalLM.register(MiniDeepSeekConfig, ToyDeepSeekProForCausalLM)

# =========================================================
# 2. 从本地加载模型与挂载 Router 拦截钩子
# =========================================================
if __name__ == "__main__":
    print(f"🚀 加载设备: {device}")
    print(f"📦 正在从 {MODEL_DIR} 读取模型权重与词表...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    current_step_routing = {}

    def get_gate_hook(layer_idx, top_k):

        def hook(module, input, output):
            last_logits = output[-1]
            scores = torch.sqrt(F.softplus(last_logits) + 1e-6)
            weights, indices = torch.topk(scores, top_k, dim=-1)
            weights = weights / (weights.sum() + 1e-6)

            expert_weights = [0.0] * module.out_features
            for idx, w in zip(indices.tolist(), weights.tolist()):
                expert_weights[idx] = w
            current_step_routing[layer_idx] = expert_weights

        return hook

    for i, layer in enumerate(model.layers):
        layer.moe.gate.register_forward_hook(
            get_gate_hook(i, model.config.num_experts_per_tok))

    # =========================================================
    # 3. 彩色 MoE 热力图渲染函数
    # =========================================================
    def render_heatmap(generated_tokens, routing_history, num_layers,
                       n_experts):
        print("\n" + "=" * 65)
        print(
            "📊 \033[1;36mDeepSeek MoE 专家激活热力图 (Per-Token Routing Heatmap)\033[0m"
        )
        print("=" * 65)

        for l in range(num_layers):
            print(
                f"\n\033[1;33m[ Layer {l} - MLA Attention 之后的 MoE 路由分布 ]\033[0m"
            )
            header = f"{'Token':<8} | " + " | ".join(
                [f"Expert {e}" for e in range(n_experts)]) + " | Shared Exp"
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            for t_idx, token_str in enumerate(generated_tokens):
                w_list = routing_history[t_idx][l]
                row_str = f"{token_str:<8} | "
                for w in w_list:
                    if w > 0.5:
                        cell = f"\033[1;32m{w*100:4.1f}%\033[0m"  # 主激活：亮绿
                    elif w > 0.0:
                        cell = f"\033[0;36m{w*100:4.1f}%\033[0m"  # 次激活：青色
                    else:
                        cell = "\033[90m  ·  \033[0m"  # 未激活：灰点
                    row_str += f" {cell}   | "
                row_str += "\033[1;35m  100% (ON)\033[0m"
                print(row_str)

        print("\n" + "-" * 65)
        print("📈 \033[1m本轮生成专家总调用频次 (Load Summary):\033[0m")
        for l in range(num_layers):
            counts = [0] * n_experts
            total_tokens = len(generated_tokens)
            for t_idx in range(total_tokens):
                for e in range(n_experts):
                    if routing_history[t_idx][l][e] > 0:
                        counts[e] += 1
            stat_str = f"Layer {l}: " + " | ".join([
                f"E{e}: {counts[e]}/{total_tokens}次" for e in range(n_experts)
            ])
            print(f"  {stat_str}")
        print("=" * 65 + "\n")

    # =========================================================
    # 4. 终端交互主程序
    # =========================================================
    print("=" * 65)
    print("🎉 终端打字机模式 + MoE 热力图诊断系统已启动！")
    print("💡 输入问题开始体验（如：'中国的首都是哪里？'），输入 q 退出")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("User ❯ ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "exit", "quit"]:
                print("退出对话，再见！👋")
                break

            if not user_input.startswith("问："):
                prompt = f"问：{user_input} 答："
            else:
                prompt = user_input if "答：" in user_input else user_input + " 答："

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            print("DeepSeek-Mini ❯ ", end="", flush=True)

            generated_tokens = []
            routing_history = []

            with torch.no_grad():
                for _ in range(30):
                    outputs = model(input_ids)
                    next_token = torch.argmax(outputs.logits[:, -1, :],
                                              dim=-1,
                                              keepdim=True)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    step_record = {
                        l: list(current_step_routing[l])
                        for l in range(model.config.num_layers)
                    }
                    routing_history.append(step_record)

                    token_text = tokenizer.decode(next_token[0],
                                                  skip_special_tokens=True)
                    generated_tokens.append(token_text)

                    # 打字机逐字输出
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
                    time.sleep(0.04)

                    input_ids = torch.cat([input_ids, next_token], dim=-1)

            print()

            if generated_tokens:
                render_heatmap(generated_tokens, routing_history,
                               model.config.num_layers,
                               model.config.n_routed_experts)

        except KeyboardInterrupt:
            print("\n退出对话，再见！👋")
            break
