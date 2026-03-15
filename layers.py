import torch
import numpy as np
import torch_mlir
from torch_mlir.ir import (
    Location, DenseElementsAttr, IntegerAttr, FloatAttr, StringAttr,
)
from typing import Optional, Union, List, Sequence, Dict

def torch_to_importer_str(dtype: torch.dtype) -> str:
    """将 PyTorch 数据类型映射为 MLIR Importer 支持的字符串。"""
    dtype_map = {
        torch.float16: "F16",
        torch.float32: "F32",
        torch.int8: "INT8",
        torch.int16: "INT16",
        torch.int32: "INT32",
        torch.bfloat16: "BF16"
    }
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise NotImplementedError(f"Unsupported dtype {dtype}")

class Tensor(object):
    """
    表示计算图中的动态数据边 (Edge / Activation)。
    """
    def __init__(self, name: str, dtype, shape: List[int], has_Parent: bool = False):
        self._name: str = name
        self.dtype = dtype
        self._shape = shape
        self._isInput = not has_Parent
        self.users: List['Layer'] = []
        self.producer: Optional['Layer'] = None

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def shape(self) -> List[int]:
        return self._shape

class parameter(object):
    """
    表示计算图中的静态参数常量 (Node Constant / Weights / Bias / RoPE tables)。
    会在 convertLayer 时被 lowering 为 hlo.ConstantOp。
    """
    def __init__(self,
                 value: Optional[Union[np.ndarray, torch.Tensor]] = None,
                 shape: Sequence[int] = None,
                 dtype = None,
                 is_buffer: bool = False):
        if value is None:
            assert isinstance(shape, (list, tuple)), f"shape must be list or tuple, got {type(shape)}"
            self._shape = list(shape)
            self._value = None
        else:
            self._shape = list(value.shape)
            self.dtype = dtype
            self._value = value
        self.is_buffer = is_buffer
        self.name: Optional[str] = None

    @property
    def shape(self) -> List[int]:
        return self._shape
    
    @property
    def value(self):
        return self._value

class Layer(object):
    """
    计算图中的计算节点 (Node / Operation)。
    """
    def __init__(self, name: str = None, outputs: List[Tensor] = None, inputs: List[Tensor] = None, params: List[parameter] = None):
        self.name: str = name
        self.outputs: List[Tensor] = outputs if outputs is not None else []
        self.inputs: List[Tensor] = inputs if inputs is not None else []
        self.params: List[parameter] = params if params is not None else []
        # 优化：显式声明残差属性，抛弃 hasattr 这种危险的动态检测
        self.has_residual: bool = False 

# Tensor A --(used by)--> Layer B --(produces)--> Tensor C
# Layer B.inputs = [Tensor A]
# Tensor A.users = [Layer B]
# Layer B.outputs = [Tensor C]
# Tensor C.producer = Layer B

    # Appends that tensor to self.inputs, meaning "this layer consumes this edge"
    # Appends self to tensor.users, meaning "this tensor is used by this layer"
    def add_inputs(self, inp: Union[Tensor, List[Tensor]]):
        inputs_to_add = [inp] if isinstance(inp, Tensor) else inp
        for tensor in inputs_to_add:
            self.inputs.append(tensor)
            tensor.users.append(self)
        
        # process Residual
        if len(self.inputs) > self.getNuminputs():
            self.has_residual = True

    # Appends the tensor to self.outputs, meaning "this layer produces this edge"
    # Sets tensor.producer = self, meaning "this tensor came from this layer"
    def add_outputs(self, outp: Union[Tensor, List[Tensor]]):
        outputs_to_add = [outp] if isinstance(outp, Tensor) else outp
        for tensor in outputs_to_add:
            self.outputs.append(tensor)
            tensor.producer = self

    def add_params(self, params: Union[parameter, List[parameter], None]):
        if params is None:
            self.params.append(None)
            return 
        params_to_add = [params] if isinstance(params, parameter) else params
        self.params.extend(params_to_add)
            
    def load_params(self, dtype, hf_w: Dict, prefix: str, key_list: Optional[Union[List[str], str]] = None):
        if key_list is None:
            keys_to_load = [""] # 表示只加载 prefix.weight 本身
        else:
            keys_to_load = [key_list] if isinstance(key_list, str) else key_list

        for p in keys_to_load:
            # 拼接正确的 key
            query = f"{prefix}.weight" if p == "" else f"{prefix}{p}.weight"
            
            if query not in hf_w:
                print(f"warning: layer {self.name} missing parameter {query}. This might be expected for some layers.")
                # 保持空占位，以对齐参数索引
                if p == "": return False 
                continue
            
            # 将 PyTorch 权重转置到正确的设备并确保存储连续性
            tensor_val = hf_w[query].to(dtype).detach().cpu().contiguous()
            self.add_params(parameter(value=tensor_val, dtype=dtype, is_buffer=False))
            
        return True
    
    def getNuminputs(self) -> int:
        raise NotImplementedError("Subclasses must implement this function")        

    def getNumparams(self) -> int:
        raise NotImplementedError("Subclasses must implement this function")
    
    def infer_output(self) -> List[int]:
        raise NotImplementedError("Subclasses must implement this function")

    def convertLayer(self, mlr_impt, sym_table):
        # 1. 拿到当前算子本该输出的 Shape 和 Type
        out_tensor = self.outputs[0]
        out_shape = tuple(out_tensor.shape) # 确保是 tuple
        
        # 获取 MLIR 侧的类型字符串 (如 "F16", "F32", "INT32")
        mlir_type_str = torch_to_importer_str(out_tensor.dtype)
        out_type = mlr_impt.mlir_type[mlir_type_str]
        
        # 2. 建立 Numpy 类型映射，防止 MLIR 构建时类型爆炸！
        np_dtype_map = {
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.float32, # numpy 原生不支持 bf16，通常用 f32 骗过去或使用第三方库
            "INT32": np.int32,
            "INT64": np.int64,
            "INT8": np.int8
        }
        np_type = np_dtype_map.get(mlir_type_str, np.float32)
        
        # 3. 动态生成对应类型和形状的全 0 假数据
        dummy_data = np.ascontiguousarray(np.zeros(out_shape, dtype=np_type))
        
        # 4. 直接硬编码 stablehlo.constant，参考 test.py 的工作路径
        result_type = mlr_impt.get_tensor_type(list(out_shape), out_type)
        raw_op = torch_mlir.ir.Operation.create(
            "stablehlo.constant",
            results=[result_type],
            attributes={
                "value": DenseElementsAttr.get(dummy_data, signless=True, type=out_type)
            },
            loc=Location.fused([Location.file(self.name, line=0, col=0)], context=mlr_impt.ctx),
        )
        mlr_impt.insert_point.insert(raw_op)

        class _OpView(object):
            def __init__(self, operation):
                self.operation = operation

        dummy_const = _OpView(raw_op)
        
        # 5. 挂载到符号表
        sym_table[out_tensor.name] = dummy_const.operation.results[0]
        
        print(f"[Dummy OP] Faked node: {self.name} | Shape: {out_shape} | Type: {mlir_type_str}")
        return dummy_const

    def basic_clone(self):
        new_layer = self.__class__.__new__(self.__class__)
        new_layer.name = self.name
        new_layer.inputs = []    
        new_layer.outputs = []    
        new_layer.params = []
        new_layer.has_residual = self.has_residual

        for p in self.params:
            if p is None:
                new_layer.params.append(None)
                continue
            new_layer.params.append(
                parameter(value=p.value.clone(), shape=p.shape, dtype=p.dtype, is_buffer=p.is_buffer)
            )
        return new_layer

# ================= 具体算子实现 =================

class Gather_embedding(Layer):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
    
    def getNuminputs(self): return 1

    def getNumparams(self) -> int: return 1
    
    def infer_output(self):
        input_tensor = self.inputs[0] 
        emb_weight = self.params[0]
        return [input_tensor.shape[0], input_tensor.shape[1], emb_weight.shape[-1]]

    def convertLayer(self, mlr_impt, sym_table):
        op = super().convertLayer(mlr_impt, sym_table)
        op.operation.attributes["hidden_size"] = IntegerAttr.get(mlr_impt.I64Type, self.hidden_size)
    
    def basic_clone(self):
        new_layer = super().basic_clone()
        new_layer.hidden_size = self.hidden_size
        return new_layer  

class Up_projection_mm(Layer):
    def __init__(self, voc_size: int):
        super().__init__()
        self.voc_size = voc_size
    
    def getNuminputs(self): return 1
    def getNumparams(self) -> int: return 1
    
    def infer_output(self):
        input_tensor = self.inputs[0] 
        emb_weight = self.params[0]
        res = list(input_tensor.shape[:-1])
        res.append(emb_weight.shape[0])
        return res
    
    def convertLayer(self, mlr_impt, sym_table):
        op = super().convertLayer(mlr_impt, sym_table)
        op.operation.attributes["voc_size"] = IntegerAttr.get(mlr_impt.I64Type, self.voc_size)
    
    def basic_clone(self):
        new_layer = super().basic_clone()
        new_layer.voc_size = self.voc_size
        return new_layer

class LlamaAttn(Layer):
    """ params: [position_ids, cos, sin, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm] """
    def __init__(self, bias: bool, drop_out: float, num_head: int, head_dim: int, num_key_value_heads: int, bs, seq_len, rope_scaling, rope_theta, dtype):
        super().__init__()
        self.attention_bias = bias
        self.drop_out = drop_out
        self.num_head = num_head
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self._generate_rotary_embedding(bs, seq_len, rope_scaling, rope_theta, dtype)
        
    def _generate_rotary_embedding(self, bs, seq_len, rope_scaling, rope_theta, dtype):
        self.rope_scaling = 1.0 if rope_scaling is None else rope_scaling
        
        # 预计算 RoPE 表并存为常量 Parameter
        theta_i = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        position_ids = torch.arange(seq_len).repeat(bs, 1).to(torch.int32)
        self.add_params(parameter(value=position_ids, dtype=torch.int32))
        
        theta_i_expand = theta_i[None, :, None].expand(position_ids.shape[0], -1, -1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        emb = (theta_i_expand @ position_ids_expanded).transpose(1, 2).contiguous()
        
        self.add_params(parameter(value=(emb.cos() * self.rope_scaling).to(dtype), dtype=dtype))
        self.add_params(parameter(value=(emb.sin() * self.rope_scaling).to(dtype), dtype=dtype))
    
    def getNuminputs(self) -> int: return 1        
    def getNumparams(self) -> int: return 9
    def infer_output(self) -> List[int]: return self.inputs[0].shape 
    
    def convertLayer(self, mlr_impt, sym_table):
        op = super().convertLayer(mlr_impt, sym_table)
        op.operation.attributes["attention_bias"] = IntegerAttr.get(mlr_impt.I64Type, self.attention_bias)
        op.operation.attributes["drop_out"] = FloatAttr.get(mlr_impt.F32Type, self.drop_out)
        op.operation.attributes["num_head"] = IntegerAttr.get(mlr_impt.I64Type, self.num_head)
        op.operation.attributes["head_dim"] = IntegerAttr.get(mlr_impt.I64Type, self.head_dim)
        op.operation.attributes["num_key_value_heads"] = IntegerAttr.get(mlr_impt.I64Type, self.num_key_value_heads)   
        return op
    
    def basic_clone(self):
        new_layer = super().basic_clone()
        new_layer.attention_bias = self.attention_bias
        new_layer.drop_out = self.drop_out
        new_layer.num_head = self.num_head
        new_layer.head_dim = self.head_dim
        new_layer.num_key_value_heads = self.num_key_value_heads
        return new_layer

class Mlp(Layer):
    def __init__(self, hidden_act: str, intermediate_size: int):
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
    
    def getNuminputs(self) -> int: return 1
    def getNumparams(self) -> int: return 3
    def infer_output(self) -> List[int]: return self.inputs[0].shape

    def convertLayer(self, mlr_impt, sym_table):
        op = super().convertLayer(mlr_impt, sym_table)
        op.operation.attributes["intermediate_size"] = IntegerAttr.get(mlr_impt.I64Type, self.intermediate_size)
        op.operation.attributes["hidden_act"] = StringAttr.get(self.hidden_act)
        return op
    
    def basic_clone(self):
        new_layer = super().basic_clone()
        new_layer.hidden_act = self.hidden_act
        new_layer.intermediate_size = self.intermediate_size
        return new_layer

class LlamaRmsNorm(Layer):
    def __init__(self, rms_norm_eps: float):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
    
    def getNuminputs(self) -> int: return 1
    def getNumparams(self) -> int: return 1
    def infer_output(self) -> List[int]: return self.inputs[0].shape
    
    def convertLayer(self, mlr_impt, sym_table):
        op = super().convertLayer(mlr_impt, sym_table)
        op.operation.attributes["rms_norm_eps"] = FloatAttr.get(mlr_impt.F32Type, self.rms_norm_eps) 
        return op
        
    def basic_clone(self):
        new_layer = super().basic_clone()
        new_layer.rms_norm_eps = self.rms_norm_eps
        return new_layer

class LlamaMatmul(Layer):
    def __init__(self, hidden_act: str, intermediate_size: int):
        super().__init__()
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size

    def getNuminputs(self) -> int: return 1
    def getNumparams(self) -> int: return 1

    def infer_output(self) -> List[int]:
        x = self.inputs[0]
        x_shape = list(x.shape)
        return list(x_shape[:-1]) + [int(self.intermediate_size)]

    def convertLayer(self, mlr_impt, sym_table):
        return super().convertLayer(mlr_impt, sym_table)

__all__ = ["LlamaRmsNorm","Mlp","LlamaAttn","Up_projection_mm","Gather_embedding","Layer","parameter","Tensor","torch_to_importer_str"]
