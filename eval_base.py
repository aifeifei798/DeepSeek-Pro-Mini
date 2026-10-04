import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入注册定义
from pretrain_turbo import MiniDeepSeekConfig, ToyDeepSeekProForCausalLM

# 可以评测最新断点，也可以评测最终成果
# MODEL_PATH = "./checkpoints/latest" if os.path.exists(
#     "./checkpoints/latest") else "./my_deepseek_pretrain_model"
MODEL_PATH = "./my_deepseek_pretrain_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📦 正在从 {MODEL_PATH} 载入待评测 Base 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


# 通用生成函数（带轻微随机性与防复读）
def complete(prompt, max_new_tokens=60, temperature=0.7, top_p=0.85):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            use_cache=False,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,  # 适度抑制复读
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# 1. 验证集困惑度 (PPL) 测试
def test_ppl(eval_text):
    inputs = tokenizer(eval_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])
        loss = outputs.loss
        ppl = torch.exp(loss)
    return loss.item(), ppl.item()


print("\n" + "=" * 65)
print("🏆 【DeepSeek-Mini Base 模型体检报告】")
print("=" * 65)

# --- 维度 1: 困惑度 (PPL) ---
test_passage = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
loss, ppl = test_ppl(test_passage)
print(f"\n📊 [1. 客观语言建模指标 (PPL)]")
print(f"  测试语段 Loss: {loss:.4f} | 困惑度 (PPL): {ppl:.2f}")

# --- 维度 2: Few-Shot 少样本理解能力 (ICL) ---
few_shot_prompt = ("问：法国的首都是哪里？ 答：巴黎。\n"
                   "问：日本的首都是哪里？ 答：东京。\n"
                   "问：中国的首都是哪里？ 答：")
print(f"\n🧠 [2. Few-Shot 上下文学习能力 (举一反三测试)]")
print(f"Prompt:\n{few_shot_prompt}")
res = complete(few_shot_prompt, max_new_tokens=15, temperature=0.2)  # 低温求稳
print(f"模型续写: \033[1;32m{res[len(few_shot_prompt):]}\033[0m")

# --- 维度 3: 常识完形填空 ---
cloze_cases = ["地球绕着太阳", "水在常温常压下的沸点是", "李白是唐朝著名的"]
print(f"\n📖 [3. 百科常识完形填空]")
for p in cloze_cases:
    res = complete(p, max_new_tokens=20, temperature=0.3)
    print(f"  输入: {p}")
    print(f"  续写: \033[1;36m{res}\033[0m\n")

# --- 维度 4: 故事生成与长文本语篇连贯性 ---
story_prompt = "从前有一只住在森林里的小松鼠，今天早上它醒来发现"
print(f"\n🎨 [4. 自由叙事与段落生成测试]")
print(f"  Prompt: {story_prompt}")
res = complete(story_prompt, max_new_tokens=80, temperature=0.75)
print(f"  模型创作故事:\n\033[1;33m{res}\033[0m")
print("\n" + "=" * 65)
