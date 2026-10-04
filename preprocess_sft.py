import os
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_PATH = "dataset/sft_t2t_mini.jsonl" if os.path.exists("dataset/sft_t2t_mini.jsonl") else "data/sft_t2t_mini.jsonl"
INPUT_BIN_PATH = "dataset/sft_inputs.bin"
LABEL_BIN_PATH = "dataset/sft_labels.bin"
MAX_SEQ_LEN = 512

print("🚀 正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
pad_id = tokenizer.pad_token_id or 0
eos_id = tokenizer.eos_token_id or 1

# 定义规范对话标记
USER_TAG = "<｜User｜>"
ASSISTANT_TAG = "<｜Assistant｜>"
EOS_TAG = "<｜end of sentence｜>"

all_input_chunks = []
all_label_chunks = []

print(f"📦 开始加工 SFT 对话数据集: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing SFT"):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            conversations = item.get("conversations", [])
            if not conversations:
                continue

            input_ids = []
            label_ids = []

            # 遍历多轮对话
            for turn in conversations:
                role = turn.get("role", "")
                content = str(turn.get("content", ""))

                if role == "user":
                    # 用户的提问：全打 -100 掩码
                    part = f"{USER_TAG}{content}{ASSISTANT_TAG}"
                    part_ids = tokenizer.encode(part, add_special_tokens=False)
                    input_ids.extend(part_ids)
                    label_ids.extend([-100] * len(part_ids))

                elif role == "assistant":
                    # 模型的回答：计算真实 Loss 并加上 EOS
                    part = f"{content}{EOS_TAG}"
                    part_ids = tokenizer.encode(part, add_special_tokens=False)
                    input_ids.extend(part_ids)
                    label_ids.extend(part_ids)  # 真实监督信号

            # 长度过滤与截断/填充
            if len(input_ids) < 8:
                continue

            if len(input_ids) > MAX_SEQ_LEN:
                input_ids = input_ids[:MAX_SEQ_LEN]
                label_ids = label_ids[:MAX_SEQ_LEN]

            # 填充到固定长度 MAX_SEQ_LEN
            pad_len = MAX_SEQ_LEN - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            label_ids = label_ids + [-100] * pad_len

            all_input_chunks.append(input_ids)
            all_label_chunks.append(label_ids)

        except Exception:
            continue

print("\n💾 正在将加工好的 SFT 数据写入极速二进制文件...")
input_arr = np.array(all_input_chunks, dtype=np.uint32)
label_arr = np.array(all_label_chunks, dtype=np.int32)  # 包含 -100，使用 int32

input_arr.tofile(INPUT_BIN_PATH)
label_arr.tofile(LABEL_BIN_PATH)

print(f"✅ SFT 数据加工圆满完成！")
print(f"📊 有效训练对话数: {len(all_input_chunks)} 条")
print(f"📁 输入张量文件: {INPUT_BIN_PATH}")
print(f"📁 标签掩码文件: {LABEL_BIN_PATH}")
