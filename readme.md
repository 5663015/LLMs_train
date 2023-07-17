# LLMs-train：一套代码微调大模型

本项目旨在微调多类基座大模型，实现 LORA + DeepSpeed + 单卡/多卡微调，目前已测试的模型见下表：

| 测试模型             | 语言 | 测试权重                                                     | 显存占用/fp16 |
| -------------------- | ---- | ------------------------------------------------------------ | ------------- |
| Chinese-LLaMA-Alpaca | 中文 | [chinese-llama-plus-lora-7b](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b) |               |
|                      |      | [chinese-llama-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-13b) |               |
|                      |      | [chinese-alpaca-plus-lora-7b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b) |               |
|                      |      | [chinese-alpaca-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b) |               |
| Open-LLaMA           | 英文 | [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b) |               |
|                      |      | [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b) |               |
| BELLE                | 中文 | [BELLE-LLaMA-EXT-13B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B) |               |
|                      |      | [BELLE-LLaMA-EXT-7B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B) |               |
| ChatGLM-6B           | 中文 | [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)        |               |
|                      |      | [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)      |               |
| 百川                 | 中文 | [baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) |               |
|                      | 中文 | [baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) |               |
| TigerBot             | 中文 | [tigerbot-7b-sft](https://huggingface.co/TigerResearch/tigerbot-7b-sft) |               |
|                      |      | [tigerbot-7b-base](https://huggingface.co/TigerResearch/tigerbot-7b-base) |               |
| Pythia               | 英文 | [pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped) |               |
|                      |      | [pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped) |               |
|                      |      | [pythia-6.9b-deduped](https://huggingface.co/EleutherAI/pythia-6.9b-deduped) |               |
|                      |      | [pythia-12b-deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped) |               |

**TODO：**

- [ ] 支持 QLoRA
- [ ] 对话界面
- [ ] 测试 Falcon
- [ ] 测试 CPM
- [ ] 添加评价指标

## Change log

- 【2023-7-？】发布第一版代码，包括LoRA+单卡/多卡微调，测试过的模型包括：Chinese-LLaMA-Alpaca、Open-LLaMA、BELLE、ChatGLM-6B、baichuan、TigerBot、Pythia。

## 运行

### 1、数据准备

这里我们使用 [CCKS2023-PromptCBLUE中文医疗大模型评测基准](https://tianchi.aliyun.com/competition/entrance/532084/introduction) 比赛中的数据集为例。此数据集将“[中文医疗信息处理挑战榜 CBLUE](https://tianchi.aliyun.com/dataset/95414?spm=a2c22.12281976.0.0.6d1746affXjGWx)”数据集进行了改造，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务，形成首个中文医疗场景的LLM评测基准。

PromptCBLUE 采用94个指令微调模板，对 CBLUE 基准中的各个任务进行。经过改造后，医疗文本 NLP 数据集都将转化为如下格式。input 字段字符串是 LLM 模型的输入，target 字段也是一个字符串，则是 LLM 模型需要生成的文本序列。其他附加信息有： type是原任务类型(不能作为模型输入)，answer_choices字段是选项，只有分类、术语标准化、推理类任务上该字段才会有实际取值，sample_id是样本编号。这些附加信息是不作为LLM的输入的。

```json
{
	"input":  str,
	"target":  str,
	"type":  str,
	"answer_choices":  str,
	"sample_id":  str,
}
```

为了方便快速验证，**我们抽取了其中的 `CHIP-CTC` 子数据集**，包括训练集 6000 条，验证集 1100 条，测试集 1060 条。



### 2、模型准备

部分 LLaMA 类的模型需要进行模型转换，涉及到的模型有：

### 3、微调

#### 环境准备

```shell
conda create -n llms_train python=3.9
conda activate llms_train
pip install -r requirements.txt
```

#### LoRA 配置

在 `configs` 文件夹里有各个模型的 LoRA 配置文件，可以自定义修改。配置文件内容举例如下：

```yaml
{
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h"
    ],
    "modules_to_save": null
}
```

字段说明：

- `lora_r`：LoRA 的秩 $r$；
- `lora_alpha`：$\frac{\alpha}{r} \Delta Wx$ 中的 $\alpha$；
- `lora_dropout`：LoRA 层的 dropout 概率；
- `lora_target_modules`：LoRA 挂在哪些 modules 上；
- `modules_to_save`：除了 LoRA 层外，还有哪些 modules 被设为可训练的，并且会被保存在最后的 checkpoint 中。

#### Deepspeed 配置

这里采用 ZeRO2 配置：

```yaml
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 100,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1e-10
    },
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

关于多卡并行训练的策略，可参考[这里](https://huggingface.co/docs/transformers/perf_train_gpu_many)。

#### 微调

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

主要参数含义如下：

- `model_name_or_path`：模型在 hugging face 上的名字，或者是已经存在本地的路径
- `use_lora`：使用 lora 微调，默认为 `True`，否则是全量微调。
- `prompt_column`：样本里输入的字段名
- `response_column`：样本里输出的字段名
- `max_source_length：`输入的最大长度
- `max_target_length`：输出的最大长度
- `pre_device_train_batch_size`：每张卡上的 batch size
- `gradient_accumulation_steps`：梯度累积轮数
- `max_steps`：训练轮数，一轮包含样本数： `GPU数量 * pre_device_train_batch_size * gradient_accumulation_steps`
- `logging_steps`：每多少轮打印 log
- `save_steps`：每多少轮保存 checkpoint
- `lora_config`：lora 配置文件路径

### 4、推理

**运行推理脚本：**


**结果示例：**


**问题记录：**

- 如果 `/work`目录没有权限，要加环境变量：`export HF_MODULES_CACHE=~/.cache/huggingface`
- sh添加权限： `chmod u+x xxx.sh`

## 代码讲解与基础知识

- [代码讲解](docs/代码讲解.md)
- [大模型基础知识](docs/大模型基础知识.md)
- [AI世界](https://www.wolai.com/rZr2VezEtFNq4fCqcdeV9D)

## 致谢

感谢社区优秀的开源大模型：[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) ([ChatGLM2](https://github.com/THUDM/ChatGLM2-6B))、[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)、[openllama](https://github.com/openlm-research/open_llama)、[BLOOM](https://huggingface.co/bigscience)、[BELLE](https://github.com/LianjiaTech/BELLE)、[Pythia](https://github.com/EleutherAI/pythia)、[GPTNeoX](https://github.com/EleutherAI/gpt-neox)、[百川](https://github.com/baichuan-inc/baichuan-7B)

此项目还参考了以下优秀的开源项目：

- [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)

## 学习交流群

## 引用

如果此项目对你有帮助，请按下面格式引用：

```latex
@software{LLMs_train,
  title = {{LLMs_train: A Set of Code to Fine-Tune Large Language Models}},
  author = {Xudong Li},
  year = {2023},
  url = {https://www.github.com/5663015/LLMs_train},
}
```
