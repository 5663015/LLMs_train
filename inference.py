import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
from peft import  PeftModel
import argparse
from tqdm import tqdm
import json
import jsonlines
from data_generate import DataGenerate
from config import CONFIG

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--data_file', type=str)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--cache_dir', type=str, default=None)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--topk', type=int, default=40)
parser.add_argument('--topp', type=int, default=0.9)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument('--repetition_penalty', type=float, default=1.2)
parser.add_argument('--max_new_tokens', type=int, default=400)
args = parser.parse_args()


generation_config = dict(
    temperature=args.temperature,
    top_k=args.topk,
    top_p=args.topp,
    do_sample=args.do_sample,
    num_beams=args.num_beams,
    repetition_penalty=args.repetition_penalty,
    max_new_tokens=args.max_new_tokens
)

sample_data = '感冒了怎么办？'


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    load_type = torch.float16
    cfg = CONFIG

    # 载入base_model
    if args.model_type in cfg.MODEL_MAP.keys():
        base_model = cfg.MODEL_MAP[args.model_type].from_pretrained(
            args.model_name_or_path,
            torch_dtype=load_type,
            trust_remote_code=True,
        ).half()
    else:
        base_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=load_type,
            trust_remote_code=True,
        ).half()
    # lora
    model = PeftModel.from_pretrained(
            base_model, 
            args.lora_path,
        ).to(device)
    model.eval()
    
    # 分词器
    if args.model_type in cfg.TOKENIZER_MAP.keys():
        tokenizer = cfg.TOKENIZER_MAP[args.model_type].from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
    # tokenizer.bos_token_id = cfg.SPECIAL_IDS[args.model_name_or_path.split('/')[-1]]['bos_id']
    # tokenizer.eos_token_id = cfg.SPECIAL_IDS[args.model_name_or_path.split('/')[-1]]['eos_id']
    # tokenizer.pad_token_id = cfg.SPECIAL_IDS[args.model_name_or_path.split('/')[-1]]['pad_id']

    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        if args.data_file.endswith('.txt'):
            with open(args.data_file, 'r') as f:
                examples = [eval(l) for l in f.readlines()]
        elif args.data_file.endswith('.jsonl') or args.data_file.endswith('.json'):
            examples = []
            with open(args.data_file, 'r') as f:
                for data in jsonlines.Reader(f):
                    examples.append(data)
        print("first 5 examples:")
        for example in examples[:5]:
            print(example)
    
    # inference
    with torch.no_grad():
        print("Start inference.")
        results = []
        for example in tqdm(examples[:2000]):
            if args.data_file.endswith('.txt'):
                input_text = example
            else:
                # input_text = example['input']
                input_text = example['instruction'][0] + example['input']

            inputs = tokenizer(input_text, return_tensors="pt")  #add_special_tokens=False ?
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device), 
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **generation_config
            )
            s = generation_output[0]
            response = tokenizer.decode(s, skip_special_tokens=True)
            response = response.replace(input_text, '')
            if args.data_file.endswith('.txt'):
                results.append({"Input":input_text, "Output":response})
            elif args.data_file.endswith('.json') or args.data_file.endswith('.jsonl'):
                example['prediction'] = response
                results.append(example)
                
        with jsonlines.open(os.path.join(args.output_path, 'generation_results.jsonl'), 'w') as writer:
            writer.write_all(results)
        with open(os.path.join(args.output_path, 'generation_config.json'), 'w') as f:
            json.dump(generation_config, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()

