'''
大模型指令微调通用代码，支持LLaMA、GLM、BLOOM基座模型
'''
import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import math
import shutil
from typing import List
from dataclasses import dataclass, field
from typing import Optional
from datasets import disable_caching
disable_caching()

import logging
import json
import torch
from transformers.utils import add_start_docstrings
import transformers
from datasets import load_dataset
import copy
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import get_last_checkpoint
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.utils import add_start_docstrings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_callback import TrainerCallback

from config import CONFIG
from arguments import TrainingArguments, ModelArguments, DataArguments
from data_generate import DataGenerate
# from data_generate_custom import DataGenerate
from utils import print_rank_0


logger = logging.getLogger(__name__)


def main():
    # ================================================================================
    # 参数、log等准备
    # ================================================================================
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    cfg = CONFIG    # 配置

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    global_rank = torch.distributed.get_rank()
    
    # 建立logging
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    log_file = os.path.join(training_args.output_dir,'print_log.txt')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    # ================================================================================
    # 建立model、分词器、LORA
    # ================================================================================
    if model_args.model_type in cfg.MODEL_MAP.keys():
        model = cfg.MODEL_MAP[model_args.model_type].from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            torchscript=model_args.torchscript
        ).half()
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            torchscript=model_args.torchscript
        ).half()

    # tokenizers
    if model_args.model_type in cfg.TOKENIZER_MAP.keys():
        tokenizer = cfg.TOKENIZER_MAP[model_args.model_type].from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
    if model_args.model_type != 'glm':
        tokenizer.bos_token_id = cfg.SPECIAL_IDS[model_args.model_name_or_path.split('/')[-1]]['bos_id']
        tokenizer.eos_token_id = cfg.SPECIAL_IDS[model_args.model_name_or_path.split('/')[-1]]['eos_id']
        tokenizer.pad_token_id = cfg.SPECIAL_IDS[model_args.model_name_or_path.split('/')[-1]]['pad_id']

    print_rank_0("tokenizer.eos_token_id = {}".format(tokenizer.eos_token_id), log_file, global_rank)
    print_rank_0("tokenizer.pad_token_id = {}".format(tokenizer.pad_token_id), log_file, global_rank)
    print_rank_0("tokenizer.bos_token_id = {}".format(tokenizer.bos_token_id), log_file, global_rank)

    # peft model
    if training_args.use_lora:
        lora_config = cfg.LORA_MAP[model_args.model_type]
        print_rank_0("Lora config: {}".format(lora_config), log_file, global_rank)
        peft_config = LoraConfig(
            r=lora_config['lora_r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['lora_target_modules'].split(','),
            modules_to_save=lora_config['modules_to_save'].split(','),
            lora_dropout=lora_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() 

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ================================================================================
    # 构建数据
    # ================================================================================
    with training_args.main_process_first(desc="loading and tokenization"):
        # data generation
        data_generator = DataGenerate(tokenizer, model_args, data_args, training_args)

        assert os.path.exists(data_args.train_file), "{} file not exists".format(data_args.train_file)
        if data_args.train_file.endswith(".json") or data_args.train_file.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_args.train_file, cache_dir=model_args.cache_dir)
        else:
            data = load_dataset(data_args.train_file, cache_dir=model_args.cache_dir)

        data.cleanup_cache_files()
        column_names = data["train"].column_names
        train_data = data["train"].shuffle().map(
            data_generator.chatglm_tokenize,
            # data_generator.chatglm_tokenize,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on train dataset",
        )
        if data_args.validation_file is not None:
            val_data = load_dataset("json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir)
            val_data = val_data["train"].shuffle().map(
                data_generator.chatglm_tokenize,
                # data_generator.chatglm_tokenize,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=False,
                num_proc=data_args.preprocessing_num_workers,
                desc="Running tokenizer on validation dataset",
            )
            print_rank_0("Eval tokenized example: ", log_file, global_rank)
            print_rank_0("input ids: {}".format(val_data[0]['input_ids']), log_file, global_rank)
            print_rank_0("inputs: {}".format(tokenizer.decode(val_data[0]['input_ids'])), log_file, global_rank)
        else:
            val_data = None

    print_rank_0("Train tokenized example: ", log_file, global_rank)
    print_rank_0("input ids: {}".format(train_data[0]['input_ids']), log_file, global_rank)
    print_rank_0("inputs: {}".format(tokenizer.decode(train_data[0]['input_ids'])), log_file, global_rank)

    # ================================================================================
    # 训练
    # ================================================================================
    #Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    print_rank_0(f"Using {training_args.half_precision_backend} half precision backend", log_file, global_rank)
  
    model.config.use_cache = False
    if training_args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save adapter_model.bin and adapter_config.json
    if training_args.use_lora:
        model.save_pretrained(training_args.output_dir)

    trainer.save_model() # https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2808
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    shutil.copyfile(
        os.path.join(training_args.output_dir, 'pytorch_model.bin'),
        os.path.join(training_args.output_dir, 'adapter_model.bin'))

    # Save model as torchscript
    if model_args.torchscript:
        traced_model = torch.jit.trace(model, 
                        [torch.tensor([train_data[0]['input_ids']]).cuda(), torch.tensor([train_data[0]['labels']]).cuda()])
        torch.jit.save(traced_model, training_args.output_dir + "/torchscript_model.pt")

    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)", log_file, global_rank)


if __name__ == "__main__":
    main()

