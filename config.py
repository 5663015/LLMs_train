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
        'bloom': BloomForCausalLM,
        'pythia': GPTNeoXForCausalLM,
        'baichuan': AutoModelForCausalLM
    }

    # 分词器
    TOKENIZER_MAP = {
        'llama': LlamaTokenizer,
        'glm': AutoTokenizer,
        'bloom': BloomTokenizerFast,
        'pythia': GPTNeoXTokenizerFast,
        'baichuan': AutoTokenizer
    }

    # bos_id & eos_id & pad_id
    SPECIAL_IDS = {
        'BelleGroup-BELLE-LLaMA-EXT-13B': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'open_llama_7b': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'open_llama_13b': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'chinese-alpaca-13b': {'bos_id': 1, 'eos_id': 2, 'pad_id': 49953},
        'chatglm2-6b': {'bos_id': None, 'eos_id': 2, 'pad_id': 0},
        'baichuan-7B': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'Baichuan-13B-Base': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'Baichuan-13B-Chat': {'bos_id': 1, 'eos_id': 2, 'pad_id': 0},
        'bloomz-7b1': {'bos_id': 1, 'eos_id': 2, 'pad_id': 3},
        'bloomz-1b7': {'bos_id': 1, 'eos_id': 2, 'pad_id': 3},
        'tigerbot-7b-sft': {'bos_id': 1, 'eos_id': 2, 'pad_id': 3},
        'tigerbot-7b-base': {'bos_id': 1, 'eos_id': 2, 'pad_id': 3},
        'pythia-12b-deduped': {'bos_id': 0, 'eos_id': 0, 'pad_id': 1},
        'pythia-12b-sft-v8-7k-steps': {'bos_id': 0, 'eos_id': 0, 'pad_id': 1},
        'pythia-1b-deduped': {'bos_id': 0, 'eos_id': 0, 'pad_id': 1},
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
        'bloom': {
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
            "lora_target_modules": "W_pack,o_proj",
            "modules_to_save": "null"
        }
    }


