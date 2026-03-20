import torch
from config import QwenConfig
from layers import *
from typing import Optional, Union, Dict, List

class GraphBuilder(object):
    """
    实现建图的基本逻辑
    """
    def __init__(self):
        self._layers: List[Layer] = []      # 对接到 MLIR 的真正的计算图
        self._inputs: List[Tensor] = []     # 模型入口结点
        self._outputs: List[Tensor] = []    # 模型出口
        self.importer = None                # 真正生成 MLIR op / module 的对象
        self.symbol_table: dict = {}        # Python对象/名字 -> MLIR SSA value

    def get_operand(self, name: str):
        if name not in self.symbol_table:
            raise KeyError(f"operand {name} not found")
        return self.symbol_table[name]

    # 建图入口: 双向引用实现的有向无环图
    def add_Layer(self, current_layer: Layer, layer_name: str, dtype) -> Tensor:
        if not (layer_name and layer_name.strip()):
            raise ValueError("Layer must have a valid name before registration.")
        current_layer.name = layer_name
        # Input(X) -> [0]Norm -> [1]Attn -> [2]Norm -> [3]Mlp
        output = Tensor(name=layer_name + '.tensor',
                        dtype=dtype,
                        shape=current_layer.infer_output(),
                        has_Parent=True)
        current_layer.add_outputs(output)
        # add next_layer at curr_layer
        self._layers.append(current_layer)
        return output

    def add_Layers(self,
                   layer_num: int,
                   layers: List[Layer], 
                   layer_keys: List[str],       # current_layer_keys
                   dtype,
                   hf_w: Dict,
                   prefix: str,     # get_prefix() => model.layers.
                   key_dict: Dict[str, List[str]],      # get_qwen_key_list()
                   mapping: Optional[List[List[int]]] = None    # PRENORM_MAPPING
                ) -> Tensor:
        output: Tensor = None

        if len(layer_keys) != len(layers):
            raise ValueError("layer_keys and layers must have the same length")

        for layer_key, sub_layer in zip(layer_keys, layers):
            # layer prefix      eg. "model.layers.3."
            query_prefix = f"{prefix}{layer_num}." if layer_num >= 0 else f"{prefix}."

            # layer_key: self_attn / mlp => "model.layers.3.self_attn"
            if layer_key not in key_dict:
                raise KeyError(f"layer key {layer_key} missing in key_dict")
            sub_layer.load_params(dtype, hf_w, query_prefix, key_dict[layer_key])

            layer_name = f"{query_prefix}{layer_key}"

            if output is not None: 
                sub_layer.add_inputs(output)
            output = self.add_Layer(sub_layer, layer_name, dtype)

        # Residual Connection: Skip Connections
        if mapping:
            for residual_connection in mapping:
                # [0, 1], [2, 3]
                if len(residual_connection) != 2:
                    raise ValueError(f"Invalid residual mapping {residual_connection}: expected [from_idx, to_idx]")

                frm = layers[residual_connection[0]]    # frm: 0,    2
                to = layers[residual_connection[1]]     #  to: 1,    3
                assert len(frm.inputs) >= 1
                # 这里获取的必须是 frm_layer 原本的输入 而不是 Norm(X)
                residual_source = frm.inputs[0]
                to.add_inputs(residual_source)
                
        return output

    def update(self, results: Union[Tensor, List[Tensor]], layers: List[Layer]):
        # results: last_outputs, layers: curr_layer
        # 当前的 Qwen3-0.6B 模型每个 layer 只有 1 个输出
        if isinstance(results, Tensor): 
            layers[0].add_inputs(results)
            return
        raise SystemError("Unsupported update routing logic")

    def convert_to_mhlo(self, model_name:str):
        from mlir_importer import MLIRImporter
        self.symbol_table.clear()
        self.importer = MLIRImporter(
            input_shapes= [i.shape for i in self._inputs],
            output_shapes= [i.shape for i in self._outputs],
            model_name=model_name,
            input_types= [torch_to_importer_str(i.dtype) for i in self._inputs],
            output_types= [torch_to_importer_str(i.dtype) for i in self._outputs],
            input_names= [i.name for i in self._inputs],
            output_names= [i.name for i in self._outputs]
        )

        # 把输入挂载到符号表
        for tensor, argument in zip(self._inputs, self.importer.func.arguments):
            register_operand(self.symbol_table, tensor.name, argument)

        for sub_layer in self._layers:
            sub_layer._convertLayer(self.importer, self.symbol_table)

        return_op = list()
        for tensor in self._outputs:
            return_op.append(self.get_operand(tensor.name))
        self.importer.create_return_op(return_op)
        return self.importer.get_module()

class qwenBuilder(GraphBuilder):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embedding: Layer = Gather_embedding(hidden_size=self.config.hidden_size)
        self.finale_norm = LlamaRmsNorm(rms_norm_eps=self.config.rms_norm_eps)
        self.up_projection: Layer = Up_projection_mm(self.config.vocab_size)
        # Router: Residual Connection
        self.PRENORM_MAPPING = [[0, 1], [2, 3]]

    def get_prefix(self) -> str:
        return "model.layers." if self.config.model_type == "qwen3" else ""

    def get_qwen_key_list(self):
        return {
            "input_layernorm": ["input_layernorm"],
            "LlamaAttn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                           "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm"],
            "post_attention_layernorm": ["post_attention_layernorm"],
            "Mlp": ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"],
        }

    def build(self, hf_w, batch_size: int = 1, seq_len: int = 288):
        # 允许动态 batch_size 和 seq_len
        dtype = self.config.torch_dtype
        prefix = self.get_prefix()
        key_dict = self.get_qwen_key_list()

        # 1. 创建入口 Tensor (没有 layer 产生 Tensor => 手动添加)
        entryTensor = Tensor(name="input_tensor",
                          dtype=torch.int32,
                          shape=[batch_size, seq_len],
                          has_Parent=False)
        self._inputs.append(entryTensor)   # Entry tensor => %arg0
        self.embedding.add_inputs(entryTensor)

        # 2. Embedding 层加载与注册
        emb_prefix = "model.embed_tokens" if self.config.model_type == "qwen3" else ""
        self.embedding.load_params(dtype, hf_w, emb_prefix, key_list=None)
        last_outputs = self.add_Layer(current_layer=self.embedding, layer_name="EmbeddingLayer", dtype=dtype)

        # 3. 循环堆叠 Transformer Decoder 层
        for i in range(self.config.num_hidden_layers):
            current_layer_keys = [
                "input_layernorm",
                "LlamaAttn",
                "post_attention_layernorm",
                "Mlp",
            ]
            current_layer_ops = [
                LlamaRmsNorm(rms_norm_eps=self.config.rms_norm_eps),
                LlamaAttn(bias=self.config.attention_bias,
                          drop_out=self.config.attention_dropout,
                          num_head=self.config.num_attention_heads,
                          head_dim=self.config.head_dim,
                          num_key_value_heads=self.config.num_key_value_heads,
                          bs=batch_size,
                          seq_len=seq_len,
                          rope_scaling=self.config.rope_scaling,
                          rope_theta=self.config.rope_theta,
                          dtype=dtype),
                LlamaRmsNorm(rms_norm_eps=self.config.rms_norm_eps),
                Mlp(hidden_act=self.config.hidden_act, intermediate_size=self.config.intermediate_size)
            ]

            self.update(last_outputs, layers=current_layer_ops)
            last_outputs = self.add_Layers(layer_num=i,
                                           layers=current_layer_ops,
                                           layer_keys=current_layer_keys,
                                           dtype=dtype,
                                           hf_w=hf_w,
                                           prefix=prefix,
                                           key_dict=key_dict,
                                           mapping=self.PRENORM_MAPPING) 

        # 4. Final Norm 连线
        self.finale_norm.load_params(dtype, hf_w, prefix="model.norm", key_list=None)
        self.update(last_outputs, layers=[self.finale_norm])
        last_outputs = self.add_Layer(current_layer=self.finale_norm, layer_name="final_norm", dtype=dtype)

        # 5. LM Head (Up Projection) 连线
        lm_head_prefix = emb_prefix if self.config.tie_word_embeddings else "lm_head"
        self.up_projection.load_params(dtype, hf_w, prefix=lm_head_prefix, key_list=None)
        self.update(last_outputs, layers=[self.up_projection])
        last_outputs = self.add_Layer(current_layer=self.up_projection, layer_name="up_projection_mm", dtype=dtype)

        # 6. 设置全局输出
        if isinstance(last_outputs, Tensor):
            self._outputs.append(last_outputs)
        else:
            self._outputs.extend(last_outputs)

model2builder:Dict[str,GraphBuilder] = {
    "qwen": qwenBuilder,
}
