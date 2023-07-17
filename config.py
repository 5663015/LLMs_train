from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
)



class CONFIG:
    # 模型
    MODEL_MAP = {
        'llama': LlamaForCausalLM,
        'glm': AutoModel,
        'bllom': BloomForCausalLM,
        'pythia': GPTNeoXForCausalLM,
        'baichuan': AutoModelForCausalLM
    }

    # 分词器
    TOKENIZER_MAP = {
        'llama': LlamaTokenizer,
        'glm': AutoTokenizer,
        'bllom': BloomTokenizerFast,
        'pythia': GPTNeoXTokenizerFast,
        'baichuan': AutoTokenizer
    }

    # LORA配置
    LORA_MAP = {
        'llama': {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj",
            "modules_to_save": "null"
        },
        'glm': {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
            "modules_to_save": "null"
        },
        'bllom': {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "query_key_value",
            "modules_to_save": "null"
        },
        'pythia': {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "query_key_value",
            "modules_to_save": "null"
        },
        'baichuan': {
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target_modules": "query_key_value",
            "modules_to_save": "null"
        }
    }


