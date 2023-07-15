import copy

IGNORE_INDEX = -100

class DataGenerate:
	def __init__(self, tokenizer, training_args):
		self.tokenizer = tokenizer
		self.training_args = training_args

	# commonly used
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
				input_ids += [self.tokenizer.eos_token_id]#make sure eos_token_id is correct
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
	
	# used for train PromptCBLUE
	def generate_for_PromptCBLUE_train(self, 
			      data_point, 
			      max_source_length=256, 
				  max_target_length=16,
				  prompt_column='input',
				  response_column='target',
				  history_column=None,
				  prefix='',
				  ignore_pad_token_for_loss=True,
				):
		max_seq_length = max_source_length + max_target_length
		model_inputs = {
			"input_ids": [],
			"labels": [],
		}
		for i in range(len(data_point[prompt_column])):
			if data_point[prompt_column][i] and data_point[response_column][i]:
				# 指令和输出
				prompt, answer = prefix + data_point[prompt_column][i], data_point[response_column][i]
				# 分词
				a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
				b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)
				# 截断
				if len(a_ids) > max_source_length - 1:
					a_ids = a_ids[: max_source_length - 1]
				if len(b_ids) > max_target_length - 2:
					b_ids = b_ids[: max_target_length - 2]

				input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

				context_length = input_ids.index(self.tokenizer.bos_token_id)
				mask_position = context_length - 1
				labels = [-100] * context_length + input_ids[mask_position+1:]
				
				pad_len = max_seq_length - len(input_ids)
				input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
				labels = labels + [self.tokenizer.pad_token_id] * pad_len

				if ignore_pad_token_for_loss:
					labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

				model_inputs["input_ids"].append(input_ids)
				model_inputs["labels"].append(labels)

		return model_inputs
	
	# used for test PromptCBLUE
	def generate_for_PromptCBLUE_test(self, 
			      data_point, 
			      max_source_length=256, 
				  max_target_length=16,
				  prompt_column='input',
				  response_column='target',
				  history_column=None,
				  prefix='',
				  ignore_pad_token_for_loss=True,
				):
		inputs, targets = [], []
		for i in range(len(data_point[prompt_column])):
			if not data_point[response_column][i]:
				targets.append("filled in !")
			else:
				targets.append(data_point[response_column][i])

			if data_point[prompt_column][i]:
				query = data_point[prompt_column][i]
				if history_column is None or len(data_point[history_column][i]) == 0:
					prompt = query
				else:
					prompt = ""
					history = data_point[history_column][i]
					for turn_idx, (old_query, response) in enumerate(history):
						prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
					prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
				inputs.append(prompt)

		inputs = [prefix + inp for inp in inputs]
		model_inputs = self.tokenizer(inputs,
									return_tensors="pt",
									max_length=max_source_length,
									truncation=True,
									padding=True)
		labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

		if ignore_pad_token_for_loss:
			labels["input_ids"] = [
				[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
			]
		model_inputs["labels"] = labels["input_ids"]

		return model_inputs


# 可以自定义DataGenerate的方法：
'''
class DataGenerate:
	def __init__():
		pass
		
	def your_generate_and_tokenize_prompt():
		...
'''



