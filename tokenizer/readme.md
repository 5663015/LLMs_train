# 分词器训练

准备 txt 格式的语料，运行命令训练分词器：

```shell
python sentencepiece_train.py \
	--model_name your_model_name \
	--corpus_name_or_path /path/to/corpus.txt \
	--output_path path/to/save \
	--vocab_size 30000 \
	--user_defined_symbols your_custom_symbols
```

参数含义如下：

- `model_name`：训练好的分词器的名字
- `corpus_name_or_path`：可以是文件夹名，这样会将此文件夹中的 txt 全拿过来训练；也可以是某个或多个语料的完整路径，多个文件的话用逗号分隔
- `output_path`：保存分词器的路径
- `vocab_size`：希望训练后的词库大小
- `user_defined_symbols`：自定义的特殊符号，多个符号用逗号分隔

训练完成后会在 `output_path` 中保存 `your_model_name.model` 文件和 `your_model_name.vocab` 文件。

训练好的新分词器可以和基座模型的分词器合并，运行下面命令：

```shell
python merge_tokenizers.py \
	--base_tokenizer_dir  your_base_tonkenizer_dir\
	--chinese_sp_model_file_or_path ./custom \
	--output_dir output
```

参数含义：

- `base_tokenizer_dir`：基座模型分词器所在路径 
- `chinese_sp_model_file_or_path`：可以是一个路径，这样会将此路径下所有的 `.model` 进行合并；也可以是一个分词器的完整路径
- `output_dir`：合并后的分词器的保存路径

代码中以合并 LLaMA 分词器为例，读者可自行修改 base tokenizer。