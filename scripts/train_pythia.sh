lora_config="lora_config_pythia_12b"
LR=2e-4
model_name_or_path="../models/pythia-12b-sft-v8-7k-steps"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/IMCS-DAC/datasets"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup torchrun --nproc_per_node 7 train.py \
    --deepspeed configs/ds_zero2_no_offload.json \
    --model_name_or_path $model_name_or_path \
    --model_type pythia \
    --use_lora True \
    --train_file $your_data_path/train_instruction_2.json \
    --validation_file $your_data_path/dev_instruction_2.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --output_dir $your_checkpopint_path/yitushibie-pythia-12b-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_steps 350 \
    --logging_steps 10 \
    --save_steps 50 \
    --learning_rate $LR \
    --fp16 \
    --lora_config configs/${lora_config}.json \
        > experiments/outputs/yitushibie-pythia-12b-lora-$LR/log 2>&1 &


# model_name_or_path='checkpoints/BelleGroup-BELLE-LLaMA-EXT-13B'
# train_file='data/instruction_NLPEC/train_result.json'
# validation_file='data/instruction_NLPEC/dev_result.json'
# cutoff_len=512
# output_dir='experiments/BelleGroup-BELLE-LLaMA-EXT-13B'

# CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 nohup torchrun --nproc_per_node 6 train/src/train.py \
#     --model_name_or_path ${model_name_or_path} \
#     --llama \
#     --use_lora True \
#     --use_int8_training \
#     --lora_config train/configs/lora_config_llama.json \
#     --train_file ${train_file} \
#     --validation_file ${validation_file} \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 2 \
#     --model_max_length ${cutoff_len} \
#     --save_strategy "steps" \
#     --save_total_limit 3 \
#     --learning_rate 5e-5 \
#     --weight_decay 0.00001 \
#     --warmup_ratio 0.05 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --evaluation_strategy "steps" \
#     --fp16 True \
#     --seed 1234 \
#     --gradient_checkpointing True \
#     --output_dir ${output_dir} > experiments/outputs2/log 2>1 &