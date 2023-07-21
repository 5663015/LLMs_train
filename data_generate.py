import copy

IGNORE_INDEX = -100

class DataGenerate:
	def __init__(self, tokenizer, model_args, data_args, training_args):
		self.tokenizer = tokenizer
		self.model_args = model_args
		self.data_args = data_args
		self.training_args = training_args

	# commonly used 1
	def generate_and_tokenize_prompt(self, data_point):
		input_ids = []
		labels = []
		source = data_point["conversations"]
		for sentence in source:
			sentence_from = sentence["from"].lower()
			sentence_value = 'Human: \n' + sentence["value"] + '\n\nAssistant: \n' if sentence_from == 'human' else sentence["value"] #https://github.com/LianjiaTech/BELLE/issues/337
			# conversation += sentence_value
			sentence_ids = self.tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
			label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
			input_ids += sentence_ids
			labels += label
			# add eos at every end of assistant sentence
			if sentence_from != 'human':
				input_ids += [self.tokenizer.eos_token_id]	# make sure eos_token_id is correct
				labels += [self.tokenizer.eos_token_id]

		input_ids = input_ids[:self.training_args.model_max_length-1]
		labels = labels[:self.training_args.model_max_length-1]
		if not any(x > -100 for x in labels):
			labels[18:24] = input_ids[18:24]#labels can not have all values being -100. 18 and 24 are just random numbers

		attention_mask = [1] * len(input_ids)
		tokenized_full_prompt = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels
		}
		return tokenized_full_prompt
	
	# commonly used 2
	def chatglm_tokenize(self, data_point):
		model_inputs = {
			"input_ids": [],
			"labels": [],
		}
		for i in range(len(data_point[self.data_args.prompt_column])):
			prompt, answer = data_point[self.data_args.prompt_column][i], \
					data_point[self.data_args.response_column][i]
			a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
			b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

			# xxx <eos>
			if len(a_ids) > self.data_args.max_source_length - 1:
				a_ids = a_ids[: self.data_args.max_source_length - 1]
			# <bos> xxx <eos>
			if len(b_ids) > self.data_args.max_target_length - 2:
				b_ids = b_ids[: self.data_args.max_target_length - 2]

			bos_token_id = 0 if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
			input_ids = a_ids + [self.tokenizer.eos_token_id, bos_token_id] + \
						b_ids + [self.tokenizer.eos_token_id]
			pad_len = self.data_args.max_source_length + self.data_args.max_target_length - len(input_ids)
			input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
			labels = [IGNORE_INDEX] * (len(a_ids) + 1) + input_ids[len(a_ids) + 1:]

			model_inputs["input_ids"].append(input_ids)
			model_inputs["labels"].append(labels)
		return model_inputs
	
	# used for train PromptCBLUE
	# def generate_for_PromptCBLUE_train(self, 
	# 		      data_point, 
	# 		      max_source_length=256, 
	# 			  max_target_length=16,
	# 			  prompt_column='input',
	# 			  response_column='target',
	# 			  history_column=None,
	# 			  prefix='',
	# 			  ignore_pad_token_for_loss=True,
	# 			):
	# 	max_seq_length = max_source_length + max_target_length
	# 	model_inputs = {
	# 		"input_ids": [],
	# 		"labels": [],
	# 	}
	# 	for i in range(len(data_point[prompt_column])):
	# 		if data_point[prompt_column][i] and data_point[response_column][i]:
	# 			# 指令和输出
	# 			prompt, answer = prefix + data_point[prompt_column][i], data_point[response_column][i]
	# 			# 分词
	# 			a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
	# 			b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
	# 			# 截断
	# 			if len(a_ids) > max_source_length - 1:
	# 				a_ids = a_ids[: max_source_length - 1]
	# 			if len(b_ids) > max_target_length - 2:
	# 				b_ids = b_ids[: max_target_length - 2]

	# 			input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

	# 			context_length = input_ids.index(self.tokenizer.bos_token_id)
	# 			mask_position = context_length - 1
	# 			labels = [-100] * context_length + input_ids[mask_position+1:]
				
	# 			pad_len = max_seq_length - len(input_ids)
	# 			input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
	# 			labels = labels + [self.tokenizer.pad_token_id] * pad_len

	# 			if ignore_pad_token_for_loss:
	# 				labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

	# 			model_inputs["input_ids"].append(input_ids)
	# 			model_inputs["labels"].append(labels)

	# 	return model_inputs
	
	# # used for test PromptCBLUE
	# def generate_for_PromptCBLUE_test(self, 
	# 		      data_point, 
	# 		      max_source_length=256, 
	# 			  max_target_length=16,
	# 			  prompt_column='input',
	# 			  response_column='target',
	# 			  history_column=None,
	# 			  prefix='',
	# 			  ignore_pad_token_for_loss=True,
	# 			):
	# 	inputs, targets = [], []
	# 	for i in range(len(data_point[prompt_column])):
	# 		if not data_point[response_column][i]:
	# 			targets.append("filled in !")
	# 		else:
	# 			targets.append(data_point[response_column][i])

	# 		if data_point[prompt_column][i]:
	# 			query = data_point[prompt_column][i]
	# 			if history_column is None or len(data_point[history_column][i]) == 0:
	# 				prompt = query
	# 			else:
	# 				prompt = ""
	# 				history = data_point[history_column][i]
	# 				for turn_idx, (old_query, response) in enumerate(history):
	# 					prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
	# 				prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
	# 			inputs.append(prompt)

	# 	inputs = [prefix + inp for inp in inputs]
	# 	model_inputs = self.tokenizer(inputs,
	# 								return_tensors="pt",
	# 								max_length=max_source_length,
	# 								truncation=True,
	# 								padding=True)
	# 	labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

	# 	if ignore_pad_token_for_loss:
	# 		labels["input_ids"] = [
	# 			[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
	# 		]
	# 	model_inputs["labels"] = labels["input_ids"]

	# 	return model_inputs


# 可以自定义DataGenerate的方法：
'''
class DataGenerate:
	def __init__():
		pass
		
	def your_generate_and_tokenize_prompt():
		...
'''


if __name__ == "__main__":
	import os
	from transformers import AutoTokenizer, LlamaTokenizer
	from config import CONFIG
	import warnings
	warnings.filterwarnings("default")

	text = "找出归一后的标准词：\n肝脏肿物切除术后\n选项：手术后胸腔积液，心脏术后，玻璃体切除术后视网膜脱离 \
				\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词"
	for model_type, models in {
		# 'llama': ['open_llama_13b', 'open_llama_7b', 'chinese-alpaca-13b'],
		'glm': ['chatglm2-6b'],
		# 'bloom': ['bloomz-7b1', 'bloomz-1b7', 'tigerbot-7b-sft', 'tigerbot-7b-base'],
		# 'pythia': ['pythia-12b-deduped', 'pythia-12b-sft-v8-7k-steps', 'pythia-1b-deduped'],
		# 'baichuan': ['Baichuan-13B-Base', 'Baichuan-13B-Chat', 'baichuan-7B'],

	}.items():
		for model in models:
			print('=' * 100)
			print(model_type, model)
			tokenizer = CONFIG.TOKENIZER_MAP[model_type].from_pretrained(
					os.path.join('../models', model),
					trust_remote_code=True
				)
			print('bos: {}, eos: {}, pad: {}'.format(tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token))
			print('bos_id: {}, eos_id: {}, pad_id: {}'.format(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id))

			ids = tokenizer.encode(text=text, add_special_tokens=False)
			print('分词后的id：', ids)
			print('文本长度：{}，分词后ids长度：{}'.format(len(text), len(ids)))
			print('ids解码后：', tokenizer.decode(ids))

# open_llama_13b: 				bos_id: 1, 	  eos_id: 2, pad_id: None
# open_llama_7b: 				bos_id: 1, 	  eos_id: 2, pad_id: None
# Chinese-LLaMA-7B: 			bos_id: 1,    eos_id: 2, pad_id: None
# chinese-alpaca-13b			bos_id: 1, 	  eos_id: 2, pad_id: 49953
# tigerbot-7b-sft				bos_id: 1, 	  eos_id: 2, pad_id: 3
# tigerbot-7b-base				bos_id: 1, 	  eos_id: 2, pad_id: 3
# chatglm2-6b: 					bos_id: None, eos_id: 2, pad_id: 0
# pythia-12b-deduped:			bos_id: 0, 	  eos_id: 0, pad_id: None
# pythia-12b-sft-v8-7k-steps: 	bos_id: 0, 	  eos_id: 0, pad_id: 1
# pythia-1b-deduped:			bos_id: 0, 	  eos_id: 0, pad_id: None
# Baichuan-13B-Base:			bos_id: 1, 	  eos_id: 2, pad_id: 0
# Baichuan-13B-Chat:			bos_id: 1, 	  eos_id: 2, pad_id: 0
# baichuan-7B:					bos_id: 1, 	  eos_id: 2, pad_id: None
# bloomz-7b1: 					bos_id: 1, 	  eos_id: 2, pad_id: 3
# bloomz-1b7: 					bos_id: 1, 	  eos_id: 2, pad_id: 3