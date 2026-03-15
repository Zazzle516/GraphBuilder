# Local Graph Builder

本地在 torch-mlir 的基础上跑通前端建图的流程

# Env

```sh
pip install torch torch-mlir iree-compiler iree-runtime 

# 可能遇到 torch-mlir 的报错
pip install --pre torch-mlir torchvision -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

安装好的环境如下

```sh
(GraphBuilder) zazzle@Zazzle-Laptop:~/GraphBuilder$ pip list
Package                  Version       Build
------------------------ ------------- -----
cuda-bindings            12.9.4
cuda-pathfinder          1.4.2
filelock                 3.25.2
fsspec                   2026.2.0
iree-compiler            20241104.1068
iree-runtime             20241104.1068
Jinja2                   3.1.6
MarkupSafe               3.0.3
mpmath                   1.3.0
networkx                 3.6.1
numpy                    2.4.3
nvidia-cublas-cu12       12.8.4.1
nvidia-cuda-cupti-cu12   12.8.90
nvidia-cuda-nvrtc-cu12   12.8.93
nvidia-cuda-runtime-cu12 12.8.90
nvidia-cudnn-cu12        9.10.2.21
nvidia-cufft-cu12        11.3.3.83
nvidia-cufile-cu12       1.13.1.3
nvidia-curand-cu12       10.3.9.90
nvidia-cusolver-cu12     11.7.3.90
nvidia-cusparse-cu12     12.5.8.93
nvidia-cusparselt-cu12   0.7.1
nvidia-nccl-cu12         2.27.5
nvidia-nvjitlink-cu12    12.8.93
nvidia-nvshmem-cu12      3.4.5
nvidia-nvtx-cu12         12.8.90
packaging                26.0
pillow                   12.1.1
pip                      26.0.1
safetensors              0.7.0
setuptools               82.0.1
sympy                    1.14.0
torch                    2.10.0        3
torch-mlir               20260314.751
torchvision              0.25.0
triton                   3.6.0
typing_extensions        4.15.0
```

安装之后先跑一下 `test.py` 看看环境有没有问题
```sh
[Frontend] MLIR 生成完毕。
[Compiler] 正在编译 MLIR 到 CPU 后端 (llvm-cpu)...
[Runtime] 正在初始化 IREE Runtime 引擎...
[Runtime] 开始执行推理计算...
运行成功！输出的结果如下：
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

# Build Graph

[Feishu-Manual Graph Construction](https://famfc6p4iwg.feishu.cn/docx/ODoNdh2PtoeAizx0fJHclEgZnzf)
