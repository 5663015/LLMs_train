import numpy as np
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json, os

from data_generate import DataGenerate


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--ckpt_path', type=str, required=True)
parser.add_argument('--use_lora', action="store_true")
parser.add_argument('--llama', action="store_true")
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--cache_dir', type=str, default=None)
args = parser.parse_args()


max_new_tokens = 512
generation_config = dict(
    temperature=0.001,
    top_k=30,
    top_p=0.85,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.2,
    max_new_tokens=max_new_tokens
)


if __name__ == '__main__':
    load_type = torch.float16 #Sometimes may need torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    # Tokenizer
    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    # model config
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # model or lora_model
    if args.use_lora:
        if args.llama:
            base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                            torch_dtype=load_type,
                                                            trust_remote_code=True
                                                            )
        else:
            base_model = AutoModel.from_pretrained(args.model_name_or_path, 
                                                            torch_dtype=load_type,
                                                            trust_remote_code=True
                                                            )
        model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, 
                                                     torch_dtype=load_type, 
                                                     config=model_config,
                                                     trust_remote_code=True
                                                    )

    if device==torch.device('cpu'):
        model.float()

    model.to(device)
    model.eval()
    print("Load model successfully")

    # dataset
    assert os.path.exists(args.data_file), "{} file not exists".format(args.data_file)
    if args.data_file.endswith(".json") or args.data_file.endswith(".jsonl"):
        test_data = load_dataset("json", data_files=args.data_file, cache_dir=args.cache_dir)
    else:
        test_data = load_dataset(args.data_file, cache_dir=args.cache_dir)
    test_data.cleanup_cache_files()
    column_names = test_data["train"].column_names

    data_generater = DataGenerate(tokenizer)

    # test_data = test_data["train"].shuffle().map(
    #     data_generater.generate_for_PromptCBLUE_test,
    #     batched=True,
    #     remove_columns=column_names,
    #     load_from_cache_file=False,
    #     desc="Running tokenizer on validation dataset",
    # )

    with torch.no_grad():
        for i, data in enumerate(test_data):
            inputs = tokenizer(data['input'], max_length=max_new_tokens, truncation=True, return_tensors="pt")
            # inputs = data["input_ids"]
            generation_output = model.generate(
                input_ids = inputs['input_ids'].to(device), 
                **generation_config
            )[0]

            generate_text = tokenizer.decode(generation_output, skip_special_tokens=True)
            print('[{}/{}]----------'.format(i+1, len(test_data)))
            print(generate_text)
