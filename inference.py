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
    # load_type = torch.float16 #Sometimes may need torch.float32
    # if torch.cuda.is_available():
    #     device = torch.device(0)
    # else:
    #     device = torch.device('cpu')

    # # Tokenizer
    # if args.model_type == 'llama':
    #     tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # if args.model_type == 'pythia':
    #     tokenizer.eos_token_id = 2
    #     tokenizer.bos_token_id = 1
    #     tokenizer.pad_token_id = 0
    # elif args.model_type == 'llama':
    #     tokenizer.pad_token_id = 0
    #     tokenizer.bos_token_id = 1
    #     tokenizer.eos_token_id = 2
    #     tokenizer.padding_side = "left"

    # # model config
    # model_config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # # model or lora_model
    # if args.use_lora:
    #     if args.model_type in ['llama', 'pythia']:
    #         base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
    #                                                         torch_dtype=load_type,
    #                                                         trust_remote_code=True
    #                                                         )
    #     elif args.model_type == 'glm':
    #         base_model = AutoModel.from_pretrained(args.model_name_or_path, 
    #                                                         torch_dtype=load_type,
    #                                                         trust_remote_code=True
    #                                                         )
    #     model = PeftModel.from_pretrained(base_model, args.ckpt_path, torch_dtype=load_type)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, 
    #                                                  torch_dtype=load_type, 
    #                                                  config=model_config,
    #                                                  trust_remote_code=True
    #                                                 )

    # if device==torch.device('cpu'):
    #     model.float()

    # model.to(device)
    # model.eval()
    # print("Load model successfully")

    # # dataset
    # assert os.path.exists(args.data_file), "{} file not exists".format(args.data_file)
    # if args.data_file.endswith(".json") or args.data_file.endswith(".jsonl"):
    #     test_data = load_dataset("json", data_files=args.data_file, cache_dir=args.cache_dir)
    # else:
    #     test_data = load_dataset(args.data_file, cache_dir=args.cache_dir)
    # test_data.cleanup_cache_files()
    # column_names = test_data["train"].column_names

    # data_generater = DataGenerate(tokenizer)

    # # test_data = test_data["train"].shuffle().map(
    # #     data_generater.generate_for_PromptCBLUE_test,
    # #     batched=True,
    # #     remove_columns=column_names,
    # #     load_from_cache_file=False,
    # #     desc="Running tokenizer on validation dataset",
    # # )

    # predictions = []
    # with torch.no_grad():
    #     for i, data in tqdm(enumerate(test_data['train'])):
    #         input_text = data['instruction'][0] + data['input']
    #         inputs = tokenizer.encode(
    #             input_text, 
    #             max_length=max_new_tokens, 
    #             truncation=True, 
    #             return_tensors="pt"
    #             )
    #         generation_output = model.generate(
    #             input_ids = inputs.to(device),
    #             **generation_config
    #         )[0]

    #         generate_text = tokenizer.decode(generation_output, skip_special_tokens=True)
    #         print('[{}/{}]----------'.format(i+1, len(test_data['train'])))
    #         print('input: \n', input_text)
    #         print('output: \n', generate_text)
    #         try:
    #             predictions.append(generate_text.split('答:')[1])
    #         except:
    #             predictions.append(generate_text)
    #         if i == 100:
    #             break

    # # 以下用于PromptCLBUE测试
    # list_test_samples = []
    # with open(args.data_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = json.loads(line)
    #         list_test_samples.append(line)
    # output_prediction_file = os.path.join(args.ckpt_path, "test_predictions.json")
    # with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #     for idx, p in enumerate(predictions):
    #         samp = list_test_samples[idx]
    #         samp["target"] = p
    #         res = json.dumps(samp, ensure_ascii=False)
    #         writer.write(f"{res}\n")
