import numpy as np
from torch_mlir.dialects import func
from torch_mlir.ir import (
    ArrayAttr,
    BF16Type,
    Context,
    F16Type,
    F32Type,
    F64Type,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    NoneType,
    RankedTensorType,
    StringAttr,
)

MLIR_TYPE_TO_NUMPY = {
    "INT8": np.int8,
    "UINT8": np.uint8,
    "SINT8": np.int8,
    "INT16": np.int16,
    "UINT16": np.uint16,
    "INT32": np.int32,
    "UINT32": np.uint32,
    "INT64": np.int64,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    # NumPy bf16 support is inconsistent across builds, so keep a safe fallback.
    "BF16": np.float32,
}

class MLIRImporter(object):
    def __init__(
        self,
        input_shapes: list,
        output_shapes: list,
        model_name: str,
        input_types: list,
        output_types: list,
        input_names: list = (),
        output_names: list = (),
    ):
        if not model_name:
            raise ValueError("model_name must not be empty")

        self.model_name = model_name
        self.input_shapes = list(input_shapes)
        self.output_shapes = list(output_shapes)
        self.num_input = len(self.input_shapes)
        self.num_output = len(self.output_shapes)

        self.ctx = Context()
        self.ctx.__enter__()
        self._ctx_entered = True
        self.loc = Location.unknown(self.ctx)
        self.loc.__enter__()
        self._loc_entered = True

        self.mlir_module = None     # MLIR.Module
        self.func = None            # func.FuncOp()
        self.entry_block = None     # first block of func.FuncOp()
        self.insert_point = None

        self.F32Type = F32Type.get()
        self.I64Type = IntegerType.get_signless(64)
        self.mlir_type = {
            "INT8": IntegerType.get_signless(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signless(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signless(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": self.F32Type,
            "F16": F16Type.get(),
            "BF16": BF16Type.get(),
        }
        self.numpy_type = MLIR_TYPE_TO_NUMPY

        if not input_names:
            input_names = [f"anonymous_input_{i}" for i in range(self.num_input)]
        if not output_names:
            output_names = [f"anonymous_output_{i}" for i in range(self.num_output)]
        self.declare_func(input_types, output_types, input_names, output_names)

    def __del__(self):
        if getattr(self, "_loc_entered", False) and getattr(self, "loc", None) is not None:
            self.loc.__exit__(None, None, None)
            self.loc = None
            self._loc_entered = False
        if getattr(self, "_ctx_entered", False) and getattr(self, "ctx", None) is not None:
            self.ctx.__exit__(None, None, None)
            self.ctx = None
            self._ctx_entered = False

    def _string_array_attr(self, values):
        return ArrayAttr.get([StringAttr.get(value) for value in values])

    def get_tensor_type(self, shape, element_type=None):
        if isinstance(element_type, str):
            element_type = self.mlir_type[element_type]

        if not isinstance(shape, (list, tuple)):
            raise TypeError(f"shape must be a list, tuple, or None, got {type(shape)}")

        return RankedTensorType.get(list(shape), element_type)

    def create_return_op(self, operands):
        if self.insert_point is None:
            raise RuntimeError("function has not been declared")
        # create a new MLIR operator
        return func.ReturnOp(operands, ip=self.insert_point)

    # 在全局 logging 中打印模型信息
    def print_module(self):
        return self.mlir_module.operation.get_asm(enable_debug_info=True, large_elements_limit=16)

    def declare_func(self, input_types: list, output_types: list, input_names: list, output_names: list):
        self.input_types = [
            self.get_tensor_type(shape, mlir_type)
            for shape, mlir_type in zip(self.input_shapes, input_types)
        ]
        self.output_types = [
            self.get_tensor_type(shape, mlir_type)
            for shape, mlir_type in zip(self.output_shapes, output_types)
        ]

        # 构建函数签名  对应到 IR 中的输入输出
        # function_type = (tensor<1x288xi32>) -> tensor<1x288x151936xf16>
        func_type = FunctionType.get(self.input_types, self.output_types)

        self.mlir_module = Module.create()
        # add debug info (optional)
        self.mlir_module.operation.attributes["torch.debug_module_name"] = StringAttr.get(self.model_name)

        # 将 funcOp 本身插入 Module
        with InsertionPoint(self.mlir_module.body):
            self.func = func.FuncOp(self.model_name, func_type)
        # 插入结点失效

        self.func.attributes["input_names"] = self._string_array_attr(list(input_names))
        self.func.attributes["output_names"] = self._string_array_attr(list(output_names))
        self.entry_block = self.func.add_entry_block()

        # 将插入入口更新为 funcOp 内部
        # 在 Layer.convertLayer() 中才可以把计算图中的算子插入
        self.insert_point = InsertionPoint(self.entry_block)

    # 对外返回接口
    def get_module(self):
        return self.mlir_module
