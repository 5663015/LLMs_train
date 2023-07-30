# LLMs-train：一套代码指令微调大模型

本项目旨在指令微调多类基座大模型，实现 LORA + DeepSpeed + 单卡/多卡微调，目前已测试的模型见下表：

| 测试模型             | 语言 | 测试权重                                                     |
| -------------------- | ---- | ------------------------------------------------------------ |
| Chinese-LLaMA-Alpaca | 中文 | [chinese-alpaca-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b) |
| Open-LLaMA           | 英文 | [open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b) |
|                      |      | [open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b) |
| BELLE                | 中文 | [BELLE-LLaMA-EXT-13B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B) |
|                      |      | [BELLE-LLaMA-EXT-7B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B) |
| BLLOM                | 英文 | [bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)   |
|                      |      | [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)   |
| ChatGLM-6B           | 中文 | [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)        |
|                      |      | [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)      |
| 百川                 | 中文 | [baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) |
|                      | 中文 | [baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) |
| TigerBot             | 中文 | [tigerbot-7b-sft](https://huggingface.co/TigerResearch/tigerbot-7b-sft) |
|                      |      | [tigerbot-7b-base](https://huggingface.co/TigerResearch/tigerbot-7b-base) |
| Pythia               | 英文 | [pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped) |
|                      |      | [pythia-12b-deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped) |

**TODO：**

- [ ] 支持 QLoRA
- [ ] LoRA 模型合并
- [ ] 对话界面
- [ ] 添加评价指标
- [ ] 测试其他基座模型

## Change log

- 【2023-7-31】发布第一版代码，包括LoRA+单卡/多卡微调、分词器训练，测试过的模型包括：Chinese-LLaMA-Alpaca、Open-LLaMA、BELLE、ChatGLM-6B、baichuan、TigerBot、Pythia。

## 运行

### 1、数据准备

这里我们使用 [CCKS2023-PromptCBLUE中文医疗大模型评测基准](https://tianchi.aliyun.com/competition/entrance/532084/introduction) 比赛中的数据集为例。此数据集将“[中文医疗信息处理挑战榜 CBLUE](https://tianchi.aliyun.com/dataset/95414?spm=a2c22.12281976.0.0.6d1746affXjGWx)”数据集进行了改造，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务，形成首个中文医疗场景的LLM评测基准。

PromptCBLUE 采用94个指令微调模板，对 CBLUE 基准中的各个任务进行。经过改造后，医疗文本 NLP 数据集都将转化为如下格式。input 字段字符串是 LLM 模型的输入，target 字段也是一个字符串，则是 LLM 模型需要生成的文本序列。

```json
{
	"input": str,
	"target": str,
	"type": str,
	"answer_choices": str,
	"sample_id": str,
}
```

为了方便快速验证，我们抽取了其中的 `CHIP-CTC` 子数据集，包括训练集 6000 条，验证集 1100 条，测试集 1060 条。[下载地址](https://huggingface.co/datasets/AIBoy1993/Prompt-CHIP-CTC)

### 2、模型准备

模型可以下载到本地，训练时给 `model_name_or_path` 参数传入模型所在的路径，也可以只传模型在 Hugging Face 上的名字，例如 `THUDM/chatglm-6b`，代码会自动下载模型。

部分 LLaMA 类的模型需要进行模型转换，涉及到的模型有：chinese-alpaca-plus-lora-13b，转换方法参考[这里](https://github.com/ymcui/Chinese-LLaMA-Alpaca#%E5%90%88%E5%B9%B6%E6%A8%A1%E5%9E%8B)。

### 3、环境与配置

#### 环境准备

```shell
conda create -n llms_train python=3.9
conda activate llms_train
pip install -r requirements.txt
```

#### LoRA 配置

在 `config.py` 文件里有各类模型的 LoRA 配置文件，可以自定义修改。配置文件内容举例如下：

```yaml
'glm': {
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": "query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
    "modules_to_save": "null"
},
```

字段说明：

- `lora_r`：LoRA 的秩 $r$；
- `lora_alpha`： $\frac{\alpha}{r} \Delta Wx$ 中的 $\alpha$；
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

### 其他配置

`config.py` 还有几个其他配置：`MODEL_MAP`、`TOKENIZER_MAP`、`SPECIAL_IDS`，根据 `model_type` 参数选取不同的 Model calss、Tokenizer Class，根据 `model_name_or_path` 选取特殊的 token id。`model_type` 取值及对应的模型如下：

- 取值 `llama`：可调用 [chinese-alpaca-plus-lora-13b](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-13b)、[open_llama_13b](https://huggingface.co/openlm-research/open_llama_13b)、[open_llama_7b](https://huggingface.co/openlm-research/open_llama_7b)、[BELLE-LLaMA-EXT-13B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-13B)、[BELLE-LLaMA-EXT-7B](https://huggingface.co/BelleGroup/BELLE-LLaMA-EXT-7B)、[tigerbot-7b-sft](https://huggingface.co/TigerResearch/tigerbot-7b-sft)、[tigerbot-7b-base](https://huggingface.co/TigerResearch/tigerbot-7b-base) 等 LLaMA 类模型。
- 取值 `glm`：可调用 [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b)、[ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)。
- 取值 `bloom`：可调用 [bloomz-1b7](https://huggingface.co/bigscience/bloomz-1b7)、[bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1) 等 BLOOM 类模型。
- 取值 `pythia`：可调用 [pythia-1b-deduped](https://huggingface.co/EleutherAI/pythia-1b-deduped)、[pythia-12b-deduped](https://huggingface.co/EleutherAI/pythia-12b-deduped) 等 Pythia 类模型。

### 4、微调

运行 `scripts/train.sh`。其文件内容如下：

```shell
LR=2e-4
model_name_or_path="../models/pythia-12b-deduped"   # LLM底座模型路径，或者是huggingface hub上的模型名称
model_type='pythia'
your_data_path="./datasets/PromptCBLUE"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs"  # 填入用来存储模型的路径
max_steps=100
max_source_length=256
max_target_length=16

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py \
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
    --output_dir $your_checkpopint_path/test-pythia-12b-deduped-lora-$LR \
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
```

参数含义如下：

- `deepspeed`：deepspeed 的配置文件路径
- `do_train`：bool，是否开启训练
- `do_eval`：bool，是否在验证集上验证，如果 `evaluation_strategy` 不等于 "no" ，则会设为 `True`
- `model_name_or_path`：模型在 hugging face 上的名字，或者是已经存在本地的路径
- `model_type`：model 的类型，可选项包括 `llama`、`glm`、` bloom`、 `pythia`、`baichuan`、` other`
- `use_lora`：使用 lora 微调，默认为 `True`，否则是全量微调
- `fp16`：是否使用 fp16 （混合）精度来训练
- `train_file`：训练集数据文件
- `validation_file`：验证集数据文件
- `preprocessing_num_workers`：在对数据进行批量分词时的 worker 数
- `cache_dir`：HF 模型的缓存路径
- `prompt_column`：样本里输入的字段名
- `response_column`：样本里输出的字段名
- `output_dir`：训练结果保存路径
- `overwrite_output_dir`：如果设为 `True`，则覆盖 output 文件夹
- `max_source_length`：输入文本的最大长度
- `max_target_length`：输出文本的最大长度
- `pre_device_train_batch_size`：训练时每张卡上的 batch size
- `pre_device_eval_batch_size`：验证/测试时每张卡上的 batch size
- `gradient_accumulation_steps`：梯度累积轮数
- `max_steps`：训练轮数，一轮包含样本数： `GPU数量 * pre_device_train_batch_size * gradient_accumulation_steps`
- `logging_steps`：每多少轮打印 log
- `save_strategy`：训过程中按照 steps 数还是 epoch 数来保存中间结果，可选值为 `no`、`steps`、`epoch`
- `save_steps`：每多少 steps 保存 checkpoint
- `evaluation_strategy`：按照 steps 数还是 epoch 数来跑验证集 ，可选值为 `no`、`steps`、`epoch`
- `eval_steps`：每多少 steps 跑一次验证
- `learning_rate`：学习率

如果是多卡训练，请对应修改 sh 中的：`CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1`。例如 4 卡训练可以改为：`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4`。

**注意：**

- `model_name_or_path` 要和 `model_type` 正确对应。
- 有些模型的 `bos_id`、`eos_id`、`pad_id` 并不是完全一致的，在 `config.py` 里的 `SPECIAL_IDS` 指定了各个模型的 special token id，除了已经测试过的模型外，需要自己手动添加。

### 5、推理

**运行推理脚本：**

```shell
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_name_or_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4 \
    --ckpt_path experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4/checkpoint-9690 \
    --model_type glm \
    --data_file ./datasets/PromptCBLUE/test.json \
    --cache_dir ./datasets/PromptCBLUE \
    --use_lora
```



**问题记录：**

- 如果 `/work`目录没有权限，要加环境变量：`export HF_MODULES_CACHE=~/.cache/huggingface`
- sh添加权限： `chmod u+x xxx.sh`

## AI 基础知识

- [大模型基础知识](docs/大模型基础知识.md)
- [AI世界](https://www.wolai.com/rZr2VezEtFNq4fCqcdeV9D)

持续更新……

## 致谢

感谢社区优秀的开源大模型：[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) ([ChatGLM2](https://github.com/THUDM/ChatGLM2-6B))、[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)、[openllama](https://github.com/openlm-research/open_llama)、[BLOOM](https://huggingface.co/bigscience)、[BELLE](https://github.com/LianjiaTech/BELLE)、[Pythia](https://github.com/EleutherAI/pythia)、[GPTNeoX](https://github.com/EleutherAI/gpt-neox)、[百川](https://github.com/baichuan-inc/baichuan-7B)。

此项目还参考了以下优秀的开源项目：

- [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)

- [sentencepiece_chinese_bpe](https://github.com/taishan1994/sentencepiece_chinese_bpe)
- [Chatglm_lora_multi-gpu](https://github.com/liangwq/Chatglm_lora_multi-gpu)
- [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning)
- [zero_nlp](https://github.com/yuanzhoulvpi2017/zero_nlp)

## 免责声明

**本项目仅供学习研究使用**。模型的训练结果受模型本身结构、随机性、训练参数、数据集等因素影响，本项目不对模型训练的结果负责，也不对模型的生成内容负责，也不对使用本项目造成的任何损失承担责任。本项目由个人在业余时间开发并维护，因投入时间有限、作者水平有限，无法保证相关问题回复的时效性，不过后续会建立交流群，到时欢迎大家一起学习、互帮互助。

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
