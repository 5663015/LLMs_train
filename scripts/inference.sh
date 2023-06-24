# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --model_name_or_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4 \
#     --ckpt_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4/checkpoint-9690 \
#     --model_type glm \
#     --data_file ./datasets/PromptCBLUE/test.json \
#     --cache_dir ./datasets/PromptCBLUE \
#     --use_lora

CUDA_VISIBLE_DEVICES=1 python inference.py \
    --model_name_or_path ../models/pythia-12b-sft-v8-7k-steps \
    --ckpt_path experiments/outputs/yitushibie-pythia-12b-lora-2e-4 \
    --model_type pythia \
    --data_file ./datasets/IMCS-DAC/datasets/test_instruction_2.json \
    --cache_dir ./datasets/IMCS-DAC/datasets \
    --use_lora
