lora_config="lora_config_pythia"
LR=2e-4
model_name_or_path="../../models/pythia-70m-deduped"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="../competitions/conversational_intention_recognition/datasets"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py \
    --model_name_or_path $model_name_or_path \
    --model_type pythia \
    --use_lora True \
    --train_file $your_data_path/train_instruction_2.json \
    --validation_file $your_data_path/dev_instruction_2.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --output_dir $your_checkpopint_path/PromptCBLUE-pythia-70m-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps 50 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --lora_config configs/${lora_config}.json 

