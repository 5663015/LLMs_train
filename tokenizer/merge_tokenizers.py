# -*- coding: UTF-8 -*-
import os
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--base_tokenizer_dir', default=None, type=str, required=True, help='原分词器文件夹')
parser.add_argument('--chinese_sp_model_file_or_path', default='./custom', type=str, help='训练好的分词器路径，文件名或者文件夹(多个分词器)')
parser.add_argument('--output_dir', default='./output', type=str, help='保存路径')
args = parser.parse_args()


chinese_sp_model_file_or_path = args.chinese_sp_model_file_or_path

# load
base_tokenizer = LlamaTokenizer.from_pretrained(args.base_tokenizer_dir)
base_spm = sp_pb2_model.ModelProto()
base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())

if os.path.isdir(chinese_sp_model_file_or_path):
    sp_models, spms = [], []
    for file in os.listdir(chinese_sp_model_file_or_path):
        if file.endswith('.model'):
            chinese_sp_model = spm.SentencePieceProcessor()
            chinese_sp_model.Load(os.path.join(chinese_sp_model_file_or_path, file))
            chinese_spm = sp_pb2_model.ModelProto()
            chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
            sp_models.append(chinese_sp_model)
            spms.append(chinese_spm)
else:
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(chinese_sp_model_file_or_path)
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())
    sp_models=  [chinese_sp_model]
    spms = [chinese_spm]


# print number of tokens
print('原分词器：')
print('长度：', len(base_tokenizer))
print('all_special_tokens: ', base_tokenizer.all_special_tokens)
print('all_special_ids: ', base_tokenizer.all_special_ids)
print('special_tokens_map: ', base_tokenizer.special_tokens_map)
print('新训练的分词器/{}个：'.format(len(spms)))
print('每个新训练的分词器长度：' + str([len(sp) for sp in sp_models]))


## Add new tokens to base tokenizer
base_spm_tokens_set = set(p.piece for p in base_spm.pieces)
print("Before: ", len(base_spm_tokens_set))
for i, spm in enumerate(spms):
    print('merge {}-th tokenizer'.format(i + 1))
    for p in spm.pieces:
        piece = p.piece
        if piece not in base_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            base_spm.pieces.append(new_p)
    base_spm_tokens_set=set(p.piece for p in base_spm.pieces)
print("New model pieces: ", len(base_spm.pieces))


## Save
output_sp_dir = os.path.join(args.output_dir, 'merged_tokenizer_sp')
output_hf_dir = os.path.join(args.output_dir, 'merged_tokenizer_hf')
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + '/merged.model', 'wb') as f:
    f.write(base_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/merged.model')
tokenizer.save_pretrained(output_hf_dir)
print("Merged tokenizer has been saved to ", output_hf_dir)


# Test
base_tokenizer = LlamaTokenizer.from_pretrained(args.base_tokenizer_dir)
merged_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text='''血清 17-OHP17-OHP 升高是 21-OHD 的特异性诊断指标和主要治疗监测指标。
一般而言，17-OHP 升高幅度越高，酶缺陷程度越重。'''
print("Test text:\n",text)
print("Tokenized by base tokenizer: ", base_tokenizer.tokenize(text))
print("Tokenized by merged tokenizer: ", merged_tokenizer.tokenize(text))
