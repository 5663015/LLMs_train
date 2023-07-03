lora_config="lora_config_pythia"
LR=2e-4
# model_name_or_path="../models/pythia-12b-sft-v8-7k-steps"   # LLM底座模型路径，或者是huggingface hub上的模型名称
# model_name_or_path="../models/neox-ckpt-pythia-1b-deduped-v1/main"
model_name_or_path="../models/pythia-70m"
your_data_path="./datasets/IMCS-DAC/datasets"  # 填入数据集所在的文件夹路径
your_checkpoint_path="./experiments/outputs"  # 填入用来存储模型的路径
# model_type="gpt-neox"
model_type="pythia"
modules_to_save="null"
peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=2 nohup torchrun --nproc_per_node 1 train.py \
    --deepspeed configs/ds_zero2_no_offload.json \
    --model_name_or_path $model_name_or_path \
    --model_type $model_type \
    --use_lora True \
    --train_file $your_data_path/train_instruction_2.json \
    --validation_file $your_data_path/dev_instruction_2.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --output_dir $your_checkpoint_path/yitushibie-pythia-70m-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps 100 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --fp16 \
    --lora_config configs/${lora_config}.json \
    --modules_to_save ${modules_to_save} \
    --torchscript \
        > log-yitushibie-pythia-70m-lora-$LR.log 2>&1 &

