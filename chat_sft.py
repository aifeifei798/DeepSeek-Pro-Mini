import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入架构定义
from pretrain_turbo import MiniDeepSeekConfig, ToyDeepSeekProForCausalLM

MODEL_DIR = "./my_deepseek_sft_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 加载设备: {device}")
print(f"📦 正在从 {MODEL_DIR} 载入最终成品的 SFT 智能对话模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
model.eval()

# 挂载 MoE 路由透视钩子
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
    layer.moe.gate.register_forward_hook(get_gate_hook(i, model.config.num_experts_per_tok))

# 彩色热力图渲染
def render_heatmap(generated_tokens, routing_history, num_layers, n_experts):
    print("\n" + "=" * 65)
    print("📊 \033[1;36mDeepSeek MoE 专家激活热力图 (Per-Token Routing)\033[0m")
    print("=" * 65)
    for l in range(num_layers):
        print(f"\n\033[1;33m[ Layer {l} 路由分布 ]\033[0m")
        header = f"{'Token':<8} | " + " | ".join([f"E{e}" for e in range(n_experts)]) + " | Shared"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for t_idx, token_str in enumerate(generated_tokens[:12]): # 打印前12个token避免刷屏
            w_list = routing_history[t_idx][l]
            row_str = f"{token_str:<8} | "
            for w in w_list:
                if w > 0.5:
                    cell = f"\033[1;32m{w*100:4.1f}%\033[0m"
                elif w > 0.0:
                    cell = f"\033[0;36m{w*100:4.1f}%\033[0m"
                else:
                    cell = "\033[90m  ·  \033[0m"
                row_str += f" {cell} | "
            row_str += "\033[1;35m 100%\033[0m"
            print(row_str)
    print("=" * 65 + "\n")

print("\n" + "=" * 65)
print("🎉 【专属 DeepSeek-Mini 智能体】正式上线！")
print("💡 支持多轮自然对话，输入 q 或 exit 退出。")
print("=" * 65 + "\n")

while True:
    try:
        user_input = input("User ❯ ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["q", "exit", "quit"]:
            print("再见！👋")
            break

        # 核心：精准对齐 SFT 训练时的对话模板
        prompt = f"<｜User｜>{user_input}<｜Assistant｜>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        print("DeepSeek-Mini ❯ ", end="", flush=True)

        generated_tokens = []
        routing_history = []

        with torch.no_grad():
            for _ in range(120): # 最多生成 120 个字
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # 🚀 工业级防复读惩罚与温度采样
                if generated_tokens:
                    for prev_token_id in set(input_ids[0].tolist()):
                        next_token_logits[0, prev_token_id] /= 1.18

                probs = F.softmax(next_token_logits / 0.65, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)

                # 双重保障：只要遇到原生 EOS 或对话结束标签，立刻收口！
                if token_id == tokenizer.eos_token_id or "<｜end of sentence｜>" in "".join(generated_tokens):
                    break

                step_record = {l: list(current_step_routing[l]) for l in range(model.config.num_layers)}
                routing_history.append(step_record)

                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                generated_tokens.append(token_text)

                sys.stdout.write(token_text)
                sys.stdout.flush()
                time.sleep(0.02)

                input_ids = torch.cat([input_ids, next_token], dim=-1)

        print()

        if len(generated_tokens) > 2:
            render_heatmap(generated_tokens, routing_history, model.config.num_layers, model.config.n_routed_experts)

    except KeyboardInterrupt:
        print("\n再见！👋")
        break
