import torch
import numpy as np
import torch_mlir
from mlir_importer import MLIR_TYPE_TO_NUMPY
from torch_mlir.ir import (
    Location, DenseElementsAttr, IntegerAttr, FloatAttr, StringAttr,
)
from typing import Optional, Union, List, Sequence, Dict

def register_operand(sym_table: Dict[str, object], name: str, operand):
    if name in sym_table:
        if sym_table[name] != operand:
            raise KeyError(f"operand {name} conflict")
    sym_table[name] = operand

TORCH_DTYPE_TO_IMPORTER = {
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "INT8",
    torch.uint8: "UINT8",
    torch.int16: "INT16",
    torch.int32: "INT32",
    torch.int64: "INT64",
    torch.bool: "BOOL",
    torch.bfloat16: "BF16",
}

def torch_to_importer_str(dtype: torch.dtype) -> str:
    if dtype in TORCH_DTYPE_TO_IMPORTER:
        return TORCH_DTYPE_TO_IMPORTER[dtype]
    raise NotImplementedError(f"Unsupported dtype {dtype}")

class Tensor(object):
    """
    表示计算图中的数据流动 (Edge / Activation)
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
    表示计算图中的权重参数常量 (Node Constant / Weights / Bias / RoPE tables)
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
    计算图中的计算节点
    """
    def __init__(self, name: str = None, outputs: List[Tensor] = None, inputs: List[Tensor] = None, params: List[parameter] = None):
        self.name: str = name
        self.outputs: List[Tensor] = outputs if outputs is not None else []
        self.inputs: List[Tensor] = inputs if inputs is not None else []
        self.params: List[parameter] = params if params is not None else []
        self.has_residual: bool = False 

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
        # 把原本独立的权重参数合并到大算子中作为参数
        if params is None:
            self.params.append(None)
            return 
        params_to_add = [params] if isinstance(params, parameter) else params
        self.params.extend(params_to_add)

    def load_params(self, dtype, hf_w: Dict, prefix: str, key_list: Optional[Union[List[str], str]] = None):
        # key_list is None: 说明 curr_layer 是单权重参数 (embedding, finale_norm, up_projection)
        # eg. ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.q_norm', 'self_attn.k_norm']
        if key_list is None:
            keys_to_load = [""]
        else:
            keys_to_load = [key_list] if isinstance(key_list, str) else key_list

        for p in keys_to_load:
            query = f"{prefix}.weight" if p == "" else f"{prefix}{p}.weight"
            # print("Qwen3-0.6B key_name:", query)
            if query not in hf_w:
                print(f"warning: layer {self.name} missing parameter {query}. This might be expected for some layers.")

            tensor_val = hf_w[query].to(dtype).detach().cpu().contiguous()
            self.add_params(parameter(value=tensor_val, dtype=dtype, is_buffer=False))
    
    def getNuminputs(self) -> int:
        raise NotImplementedError("Subclasses must implement this function")        

    def getNumparams(self) -> int:
        raise NotImplementedError("Subclasses must implement this function")

    def infer_output(self) -> List[int]:
        raise NotImplementedError("Subclasses must implement this function")

    def convertLayer(self, mlr_impt, sym_table):
        out_tensor = self.outputs[0]    # 当前模型每个 layer 都只有 1 个输出
        out_shape = tuple(out_tensor.shape)

        # 根据类型映射表获取 torch 对应到 MLIR 的 type-str  再对应到真实的 mlir_type
        mlir_type_str = torch_to_importer_str(out_tensor.dtype)     # "F16"
        mlir_out_type = mlr_impt.mlir_type[mlir_type_str]           # type(f16)

        dummy_type = mlr_impt.numpy_type.get(mlir_type_str, MLIR_TYPE_TO_NUMPY["F32"])
        dummy_data = np.ascontiguousarray(np.zeros(out_shape, dtype=dummy_type))

        # 硬编码 stablehlo.constantOp
        result_mlir_type = mlr_impt.get_tensor_type(list(out_shape), mlir_out_type) # eg. tensor<1x288x1024xf16>
        raw_op = torch_mlir.ir.Operation.create(
            "stablehlo.constant",
            results=[result_mlir_type],
            attributes={
                "value": DenseElementsAttr.get(dummy_data, signless=True, type=mlir_out_type)
            },
            loc=Location.fused([Location.file(self.name, line=0, col=0)], context=mlr_impt.ctx),
        )

        # 插入该算子
        mlr_impt.insert_point.insert(raw_op)

        # 如果需要针对不同算子提供一个通用接口  可以使用 wrapper class
        # class _OpView(object):
        #     def __init__(self, operation):
        #         self.operation = operation
        # dummy_const = _OpView(raw_op)

        # 把输入挂载到符号表
        # Q: 这个符号表是怎么起作用的
        register_operand(sym_table, out_tensor.name, raw_op.results[0])

        print(f"[Dummy OP] Faked node: {self.name} | Shape: {out_shape} | Type: {mlir_type_str}")
        return raw_op

    # Manual copy constructor hook for layer obj
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
        op.attributes["hidden_size"] = IntegerAttr.get(mlr_impt.I64Type, self.hidden_size)
        return op
    
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
        op.attributes["voc_size"] = IntegerAttr.get(mlr_impt.I64Type, self.voc_size)
        return op

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
        
        # Precompute RoPE save as Parameter
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
        op.attributes["attention_bias"] = IntegerAttr.get(mlr_impt.I64Type, self.attention_bias)
        op.attributes["drop_out"] = FloatAttr.get(mlr_impt.F32Type, self.drop_out)
        op.attributes["num_head"] = IntegerAttr.get(mlr_impt.I64Type, self.num_head)
        op.attributes["head_dim"] = IntegerAttr.get(mlr_impt.I64Type, self.head_dim)
        op.attributes["num_key_value_heads"] = IntegerAttr.get(mlr_impt.I64Type, self.num_key_value_heads)
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
        op.attributes["intermediate_size"] = IntegerAttr.get(mlr_impt.I64Type, self.intermediate_size)
        op.attributes["hidden_act"] = StringAttr.get(self.hidden_act)
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
        op.attributes["rms_norm_eps"] = FloatAttr.get(mlr_impt.F32Type, self.rms_norm_eps)
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

__all__ = ["LlamaRmsNorm","Mlp","LlamaAttn","Up_projection_mm","Gather_embedding","Layer","parameter","Tensor","torch_to_importer_str", "register_operand"]
