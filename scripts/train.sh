LR=2e-4
model_name_or_path="THUDM/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
model_type='glm'
your_data_path="./datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs"  # 填入用来存储模型的路径
max_steps=100
max_source_length=256
max_target_length=16

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=4,7 torchrun --nproc_per_node 2 --master_port 29700 train.py \
    --deepspeed configs/ds_zero2_no_offload.json \
    --do_train \
    --do_eval \
    --model_name_or_path $model_name_or_path \
    --model_type $model_type \
    --use_lora True \
    --fp16 \
    --train_file $your_data_path/train_CHIP-CTC.json \
    --validation_file $your_data_path/dev_CHIP-CTC.json \
    --preprocessing_num_workers 8 \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --output_dir $your_checkpopint_path/test-pythia-1b-deduped-lora-$LR-2 \
    --overwrite_output_dir \
    --max_source_length $max_source_length \
    --max_target_length $max_target_length \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps $max_steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --learning_rate $LR 
