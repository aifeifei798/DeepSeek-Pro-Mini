import os
import json
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_PATH = "dataset/pretrain_t2t_mini.jsonl" if os.path.exists(
    "dataset/pretrain_t2t_mini.jsonl") else "data/pretrain_t2t_mini.jsonl"
BIN_OUT_PATH = "dataset/pretrain_data.bin"

print("🚀 正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
eos_id = tokenizer.eos_token_id or 1

print(f"📦 开始离线打包 {DATA_PATH} 为二进制文件，请稍候...")
all_token_ids = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Tokenizing"):
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line).get("text", "")
            if text:
                all_token_ids.extend(tokenizer.encode(text) + [eos_id])
        except Exception:
            continue

# 转为 uint32 二进制存储（词表 12.9万，需 uint32）
arr = np.array(all_token_ids, dtype=np.uint32)
arr.tofile(BIN_OUT_PATH)

print(f"\n✅ 打包完成！总 Token 数: {len(arr) / 1e6:.2f} M (百万)")
print(f"📁 极速二进制文件已生成: {BIN_OUT_PATH}")
