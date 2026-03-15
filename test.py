import pathlib
import tempfile

import numpy as np
import iree.compiler as ireec
import iree.runtime as ireert

# 前端建图测试
# builder = qwenBuilder(hf_w, config)
# builder.build(hf_w)
# mlir_module = builder.convert_to_mhlo("qwen3")
# mlir_text = str(mlir_module)

# 测试算子
mlir_text = """
module {
  func.func public @qwen3(%arg0: tensor<1x288xi32>) -> tensor<1x288xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<1x288xi32>) -> tensor<1x288xf32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<1x288xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<1x288xf32>
    return %2 : tensor<1x288xf32>
  }
}
"""
print("[Frontend] MLIR 生成完毕。")

print("[Compiler] 正在编译 MLIR 到 CPU 后端 (llvm-cpu)...")
compiled_vmfb = ireec.compile_str(
    mlir_text,
    target_backends=["llvm-cpu"],
    extra_args=[
        "--iree-input-type=stablehlo",
        "--iree-llvmcpu-target-cpu=generic",
    ],
)

print("[Runtime] 正在初始化 IREE Runtime 引擎...")
with tempfile.TemporaryDirectory() as tmpdir:
    vmfb_path = pathlib.Path(tmpdir) / "qwen3.vmfb"
    vmfb_path.write_bytes(compiled_vmfb)
    module = ireert.load_vm_flatbuffer_file(str(vmfb_path), driver="local-task")

    # 准备输入数据
    dummy_input = np.ones((1, 288), dtype=np.int32)
    dummy_input = np.ascontiguousarray(dummy_input)

    print("[Runtime] 开始执行推理计算...")
    result = module["qwen3"](dummy_input)
    host_result = result.to_host()

    print("运行成功！输出的结果如下：")
    print(host_result)
