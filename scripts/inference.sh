CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_name_or_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4 \
    --ckpt_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4/checkpoint-9690 \
    --data_file ./datasets/PromptCBLUE/test.json \
    --cache_dir ./datasets/PromptCBLUE \
    --use_lora