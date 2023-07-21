# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --model_name_or_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4 \
#     --ckpt_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4/checkpoint-9690 \
#     --model_type glm \
#     --data_file ./datasets/PromptCBLUE/test.json \
#     --cache_dir ./datasets/PromptCBLUE \
#     --use_lora

CUDA_VISIBLE_DEVICES=3 python inference.py \
    --model_name_or_path ../models/chatglm2-6b \
    --lora_path experiments/outputs/pldccmt-45w-chatglm2-6b-lora-4e-4-1500-512-512 \
    --model_type glm \
    --data_file ../instruction/pldccmt-45w_dev.jsonl \
    --output_path experiments/outputs/pldccmt-45w-chatglm2-6b-lora-4e-4-1500-512-512
