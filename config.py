from typing import Any,Dict,Optional

class Config:
    def __init__(self,
                 architectures: str,
                 bos_token_id: int,
                 eos_token_id: int,
                 hidden_size: int,
                 initializer_range: float,
                 intermediate_size: int,
                 model_type: str,
                 num_attention_heads: int,
                 num_hidden_layers: int,
                 rms_norm_eps: float,
                 attention_bias: bool = False,
                 attention_dropout: float = 0.0,
                 head_dim: Optional[int] = None,
                 hidden_act: str = "silu",
                 max_position_embeddings: Optional[int] = None,
                 max_window_layers: Optional[int] = None,
                 num_key_value_heads:int = None, 
                 rope_scaling: Optional[Any] = None,
                 rope_theta: Optional[int] = None,
                 sliding_window: Optional[Any] = None,
                 tie_word_embeddings: Optional[bool] = None,
                 torch_dtype = None,
                 transformers_version: Optional[str] = None,
                 use_cache: bool = True,
                 use_sliding_window: Optional[bool] = None,
                 vocab_size: Optional[int] = None):
        #basic info
        self.architectures = architectures
        self.model_type = model_type
        self.sliding_window = sliding_window
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        #vocab-token info
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        #atten info
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        #Mlp info
        self.intermediate_size = intermediate_size 
        #norm info
        self.rms_norm_eps = rms_norm_eps
        #weights info
        self.initializer_range = initializer_range
        #other info
        self.max_window_layers = max_window_layers
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

    @classmethod
    def canonicalize(cls, hf_c) -> "Config":
        raise NotImplementedError(f"{cls.__name__}.canonicalize() must be implemented by subclasses.")
    
class QwenConfig(Config):
    @classmethod
    def canonicalize(cls, hf_c: dict, dtype) -> "QwenConfig":
        if hasattr(hf_c, "to_dict"):
            hf_dict = hf_c.to_dict()
        elif isinstance(hf_c, dict):    # already dict
            hf_dict = hf_c
        else:
            raise TypeError(f"Unsupported config type: {type(hf_c)}")

        # 如果 HF 配置里没有 num_key_value_heads 说明模型没有使用 QGA 优化  退化到 MHA 执行
        # (head_dim, num_key_value_heads) = (128, 8)
        head_dim = hf_dict["hidden_size"] // hf_dict["num_attention_heads"] if "head_dim" not in hf_dict.keys() else hf_dict["head_dim"]
        num_key_value_heads = hf_dict["num_attention_heads"] if "num_key_value_heads" not in hf_dict.keys() else hf_dict["num_key_value_heads"]

        return cls(
            architectures=hf_dict.get("architectures", ["QwenForCausalLM"])[0],
            attention_bias=hf_dict.get("attention_bias"),
            attention_dropout=hf_dict.get("attention_dropout"),
            bos_token_id=hf_dict.get("bos_token_id"),
            eos_token_id=hf_dict.get("eos_token_id"),
            head_dim=head_dim,
            hidden_act=hf_dict.get("hidden_act"),
            hidden_size=hf_dict.get("hidden_size"),
            initializer_range=hf_dict.get("initializer_range"),
            intermediate_size=hf_dict.get("intermediate_size"),
            max_position_embeddings=hf_dict.get("max_position_embeddings"),
            max_window_layers=hf_dict.get("max_window_layers"),
            model_type=hf_dict.get("model_type"),
            num_attention_heads=hf_dict.get("num_attention_heads"),
            num_hidden_layers=hf_dict.get("num_hidden_layers"),
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=hf_dict.get("rms_norm_eps"), 
            rope_scaling=hf_dict.get("rope_scaling"),
            rope_theta=hf_dict.get("rope_theta"),
            sliding_window=hf_dict.get("sliding_window"),
            tie_word_embeddings=hf_dict.get("tie_word_embeddings"),
            torch_dtype=dtype,
            transformers_version=hf_dict.get("transformers_version"),
            use_cache=hf_dict.get("use_cache"),
            use_sliding_window=hf_dict.get("use_sliding_window"),
            vocab_size=hf_dict.get("vocab_size"),
        )

model2config:Dict[str,Config] = {
    "qwen": QwenConfig,
    "llama": None
}
