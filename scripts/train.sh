lora_config="lora_config_chatglm_6b"
LR=2e-4
model_name_or_path="/home/lixudong39/models/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node 1 train.py \
    --model_name_or_path $model_name_or_path \
    --use_lora True \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 200 \
    --learning_rate $LR \
    --lora_config configs/${lora_config}.json \
    --modules_to_save ${modules_to_save} \
        > experiments/outputs/PromptCBLUE-chatglm-6b-lora-$LR/log 2>&1 &
