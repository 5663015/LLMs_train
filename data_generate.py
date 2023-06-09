import copy


def generate_and_tokenize_prompt(data_point):
        input_ids = []
        labels = []
        source = data_point["conversations"]
        for sentence in source:
            sentence_from = sentence["from"].lower()
            sentence_value = 'Human: \n' + sentence["value"] + '\n\nAssistant: \n' if sentence_from == 'human' else sentence["value"] #https://github.com/LianjiaTech/BELLE/issues/337
            # conversation += sentence_value
            sentence_ids = tokenizer.encode(sentence_value, add_special_tokens=False)#do not add bos_token_id
            label = copy.deepcopy(sentence_ids) if sentence_from != 'human' else [IGNORE_INDEX] * len(sentence_ids)
            input_ids += sentence_ids
            labels += label
            # add eos at every end of assistant sentence
            if sentence_from != 'human':
                input_ids += [tokenizer.eos_token_id]#make sure eos_token_id is correct
                labels += [tokenizer.eos_token_id]

        input_ids = input_ids[:training_args.model_max_length-1]
        labels = labels[:training_args.model_max_length-1]
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[18:24]#labels can not have all values being -100. 18 and 24 are just random numbers

        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt
