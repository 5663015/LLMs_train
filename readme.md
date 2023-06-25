# LLMs-train：一套代码微调大模型

本项目旨在微调多类基座大模型，实现 LORA + DeepSpeed + 单卡/多卡微调，目前已测试以下几类模型：

- LLaMA-7B
- ChatGLM-6B
- BLOOM
- Pythia-13B



**TODO：**

- [ ] 测试baichuan-7B
- [ ] 测试ChatGLM2-6B
- [ ] 支持QLoRA
- [ ] 对话界面



## 运行

### 1、数据准备

这里我们使用 [CCKS2023-PromptCBLUE中文医疗大模型评测基准](https://tianchi.aliyun.com/competition/entrance/532084/introduction) 比赛中的数据集。

### 2、微调

**环境准备：**

```shell
conda create -n llms_train python=3.9
conda activate llms_train
pip install -r requirements.txt
```

**微调脚本参数说明：**

```shell
lora_config="lora_config_chatglm_6b"
LR=2e-4
model_name_or_path="/home/lixudong39/models/chatglm-6b"   
your_data_path="./datasets/PromptCBLUE"  
your_checkpopint_path="./experiments/outputs"  
peft_path=""  

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --nproc_per_node 4 train.py \
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
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 300 \
    --learning_rate $LR \
    --lora_config configs/${lora_config}.json
```

- `model_name_or_path`：模型在 hugging face 上的名字，或者是已经存在本地的路径
- `use_lora`：使用 lora 微调，默认为 `True`，否则是全量微调。
- 





**微调ChatGLM-6B：**





### 3、推理



**结果示例：**





**问题记录：**

- 如果 `/work`目录没有权限，要加环境变量：`export HF_MODULES_CACHE=~/.cache/huggingface`
- sh添加权限： `chmod u+x xxx.sh`



## 代码讲解与基础知识

- [代码讲解](docs/代码讲解.md)
- [大模型基础知识](docs/大模型基础知识.md)
- [AI世界](https://www.wolai.com/rZr2VezEtFNq4fCqcdeV9D)

## 致谢

此代码参考了以下优秀的开源项目：

- BELLE
- 



## 引用

如果此项目对你有帮助，请按下面格式引用：

```latex
@software{LLMs_train,
  title = {{LLMs_train: A Set of Code to Fine-Tune Large Language Models}},
  author = {5663015},
  url = {https://www.github.com/5663015/LLMs_train},
}
```

