from safetensors import safe_open
import os

model_dir = "/home/zazzle/Models/Qwen3-0.6B"
weight_path = os.path.join(model_dir, 'model.safetensors')

# 使用 safe_open，只读取元数据，不加载真实 tensor
with safe_open(weight_path, framework="pt", device="cpu") as f:
    print(f"总共有 {len(f.keys())} 个权重张量。\n")
    
    for key in f.keys():
        shape = f.get_slice(key).get_shape()
        print(f"{key}  --->  {shape}")