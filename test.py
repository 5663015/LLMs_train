import jsonlines
from tqdm import tqdm


c = 0
with open('../instruction/pldccmt-45w_train.jsonl', 'r') as f:
    for data in tqdm(jsonlines.Reader(f)):
        if data['input'] and data['output']:
            c +=1
        else:
            print(data)
            break
print(c)
