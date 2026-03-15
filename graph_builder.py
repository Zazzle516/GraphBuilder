import torch
from collections import Counter
from config import QwenConfig
from layers import *
from typing import Optional, Union, Dict, List

class GraphBuilder(object):
    """
    仅负责维护 Layer 列表、输入输出 Tensor、以及调用底层 API 注册节点与连线。
    不包含任何特定模型的组装逻辑。
    """
    def __init__(self):
        self._layers: List[Layer] = []      # 对接到 MLIR 的真正的计算图
        self._inputs: List[Tensor] = []     # 模型入口结点
        self._outputs: List[Tensor] = []    # 模型出口
        self.importer = None                # 真正生成 MLIR op / module 的对象
        self.symbol_table: dict = {}        # Python对象/名字 -> MLIR SSA value

    def add_operand(self, name: str, operand):
        # operand: input Tensor
        if name in self.symbol_table:
            if self.symbol_table[name] != operand:
                raise KeyError(f"operand {name} conflict")
            return 
        self.symbol_table[name] = operand

    def get_operand(self, name: str):
        if name not in self.symbol_table:
            raise KeyError(f"operand {name} not found")
        return self.symbol_table[name]

    # 建图入口: 双向引用实现的有向无环图
    def add_Layer(self, current_layer: Layer, layer_name: str, dtype) -> Tensor:
        if not (layer_name and layer_name.strip()):
            raise ValueError("Layer must have a valid name before registration.")
        current_layer.name = layer_name
        # Input (X) -> [0]Norm -> [1]Attn -> [2]Norm -> [3]Mlp
        # 用上一层 layer 的 output 作为当前 layer 的输入
        output = Tensor(name=layer_name + '.tensor',
                        dtype=dtype,
                        shape=current_layer.infer_output(),
                        has_Parent=True)
        current_layer.add_outputs(output)
        # 再添加下一层 layer
        self._layers.append(current_layer)
        return output

    def add_Layers(self,
                   layer_num: int,
                   layers: List[Layer], 
                   dtype,
                   hf_w: Dict,
                   prefix: str,
                   key_dict: Dict,
                   mapping: Optional[List[List[int]]] = None) -> Tensor:
        output: Tensor = None
        cnt = Counter()
        
        for l in layers:
            cls = type(l)
            num = cnt[cls]
            cnt[cls] += 1
            
            query_prefix = f"{prefix}{layer_num}." if layer_num >= 0 else f"{prefix}."
            
            # 处理权重加载
            if cls in key_dict:
                if isinstance(key_dict[cls], dict):
                    if num not in key_dict[cls]:
                        raise IndexError(f"op {cls} number {num} lost weights index in key_dict") 
                    query_list = key_dict[cls][num]
                    l.load_params(dtype, hf_w, query_prefix, query_list)
                elif isinstance(key_dict[cls], list):
                    l.load_params(dtype, hf_w, query_prefix, key_dict[cls])
                else:
                    raise TypeError("type of op's key_dict unsupported yet")
            
            # 生成节点名称
            layer_name = f"{query_prefix}{l.__class__.__name__}_{num}" if layer_num >= 0 else f"{query_prefix}{l.__class__.__name__}"

            if output is not None: 
                l.add_inputs(output)
            output = self.add_Layer(l, layer_name, dtype)

        # Residual Connection: X + Attention(Norm(X))
        if mapping:
            for edge_map in mapping:
                if len(edge_map) != 2:
                    print(f"warning: edge_map-{edge_map} error")
                    continue
                frm = layers[edge_map[0]]
                to = layers[edge_map[1]]
                assert len(frm.inputs) >= 1
                # 仅支持一个输出 (idx=0) 有多个 user 的情况
                edge = frm.inputs[0]
                to.add_inputs(edge)
                
        return output

    def update(self, results: Union[Tensor, List[Tensor]], layers: List[Layer]):
        if isinstance(results, Tensor): 
            layers[0].add_inputs(results)
            return 
        if len(layers) == 1 and len(results) > 1:
            target = layers[0]
            for i, res in enumerate(results):
                if i >= target.getNuminputs():
                    raise ValueError("inputs overflow while updating")
                target.add_inputs(res)
            return 
        raise SystemError("Unsupported update routing logic")

    def print_graph(self,logger):
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~inputs~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i,input in enumerate(self._inputs):
            logger.info(f"---------------tensor:{input.name}---------------")
            logger.info(f"dtype:{input.dtype}")
            logger.info(f"shape:{input.shape}")
            logger.info(f"father:{input.producer}")
            logger.info(f"isinput:{input._isInput}")
            for j in input.users:
                logger.info(f"users:{j.name}")
            logger.info("----------------------------------------")
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~graph structure~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i,l in enumerate(self._layers):
            logger.info(f"---------------{l.name}---------------")
            for j in l.inputs:
                logger.info(f"input: {j.name}")
            logger.info(f"params: {l.params}")
            if len(l._outputs)==1:
                logger.info(f"output: {l.outputs[0].name}")
                logger.info("~~~~~~produce")
                logger.info(f"~~~~~~name:{l.outputs[0].name}")
                logger.info(f"~~~~~~shape:{l.outputs[0].shape}")
                logger.info(f"~~~~~~dtype:{l.outputs[0].dtype}")
                logger.info(f"~~~~~~father:{l.outputs[0].producer.name}")
                logger.info(f"~~~~~~isinput:{l.outputs[0]._isInput}")
                for j in l.outputs[0].users:
                    logger.info(f"~~~~~~users:{j.name}")
            logger.info("----------------------------------------")
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def convert_to_mhlo(self, model_name:str):
        """convert all to mlir"""
        from mlir_importer import MLIRImporter
        self.importer = MLIRImporter(
            input_shapes= [i.shape for i in self._inputs],
            output_shapes= [i.shape for i in self._outputs],
            model_name=model_name,
            input_types= [torch_to_importer_str(i.dtype) for i in self._inputs],
            output_types= [torch_to_importer_str(i.dtype) for i in self._outputs],
            input_names= [i.name for i in self._inputs],
            output_names= [i.name for i in self._outputs]
        )

        for idx, tensor in enumerate(self._inputs):
            self.add_operand(tensor.name, self.importer.func.arguments[idx])

        for l in self._layers:
            l.convertLayer(self.importer,self.symbol_table)

        return_opr = list()
        for idx, tensor in enumerate(self._outputs):
            opr = self.get_operand(tensor.name)
            return_opr.append(opr)
        self.importer.create_return_op(return_opr)
        return self.importer.get_module()

class qwenBuilder(GraphBuilder):
    """
    特定于 Qwen 模型的宏观构图逻辑。
    负责控制 Transformer 层数、循环、以及前后处理模块的连线。
    """
    def __init__(self, hf_w, config: QwenConfig):
        super().__init__()
        self.config = config
        self.embedding: Layer = Gather_embedding(hidden_size=self.config.hidden_size)
        self.finale_norm = LlamaRmsNorm(rms_norm_eps=self.config.rms_norm_eps)
        self.up_projection: Layer = Up_projection_mm(self.config.vocab_size)
        # Router: Residual Connection  GraphBuilder.add_Layers()
        self.POSTNORM_MAPPING = [[0, 1], [2, 3]]

    def get_prefix(self) -> str:
        return "model.layers." if self.config.model_type == "qwen3" else ""

    def get_qwen_key_list(self):
        return {
            LlamaAttn: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                        "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm"],
            Mlp: ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"],
            LlamaRmsNorm: {
                0: ["input_layernorm"],
                1: ["post_attention_layernorm"]
            }
        }

    def build(self, hf_w, batch_size: int = 1, seq_len: int = 288):
        # 允许动态 batch_size 和 seq_len
        dtype = self.config.torch_dtype
        prefix = self.get_prefix()
        key_dict = self.get_qwen_key_list()

        # 1. 创建入口 Tensor (没有 layer 产生  手动添加)
        sentence = Tensor(name="input_sentence",
                          dtype=torch.int32,
                          shape=[batch_size, seq_len],
                          has_Parent=False)
        self._inputs.append(sentence)   # 入口需要额外对接到 MLIR
        self.embedding.add_inputs(sentence)

        # 2. Embedding 层加载与注册
        emb_prefix = "model.embed_tokens" if self.config.model_type == "qwen3" else ""
        self.embedding.load_params(dtype, hf_w, emb_prefix, key_list=None)
        last_outputs = self.add_Layer(current_layer=self.embedding, layer_name="EmbeddingLayer", dtype=dtype)

        # 3. 循环堆叠 Transformer Decoder 层
        for i in range(self.config.num_hidden_layers):
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
                                           dtype=dtype,
                                           hf_w=hf_w,
                                           prefix=prefix,
                                           key_dict=key_dict,
                                           mapping=self.POSTNORM_MAPPING) 

        # 4. Final Norm 连线
        self.finale_norm.load_params(dtype, hf_w, prefix="model.norm", key_list=None)
        self.update(last_outputs, layers=[self.finale_norm])
        last_outputs = self.add_Layer(current_layer=self.finale_norm, layer_name="final_norm", dtype=dtype)   

        # 5. LM Head (Up Projection) 连线
        lm_head_prefix = emb_prefix if self.config.tie_word_embeddings else "lm_head"
        self.up_projection.load_params(dtype, hf_w, prefix=lm_head_prefix, key_list=None)
        self.update(last_outputs, layers=[self.up_projection])
        last_outputs = self.add_Layer(current_layer=self.up_projection, layer_name="up_projtion_mm", dtype=dtype)

        # 6. 设置全局输出
        if isinstance(last_outputs, Tensor):
            self._outputs.append(last_outputs)
        else:
            self._outputs.extend(last_outputs)

model2builder:Dict[str,GraphBuilder] = {
    "qwen": qwenBuilder,
}