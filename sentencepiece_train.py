import os
import argparse
import sentencepiece as spm


# https://github.com/google/sentencepiece
# https://juejin.cn/post/7234795667477561402
# https://github.com/taishan1994/sentencepiece_chinese_bpe/blob/main/train_bpe.py


parser = argparse.ArgumentParser()
parser.add_argument('--corpus_name_or_path', type=str, required=True)
parser.add_argument('--output_path', type=str, default='./tokenizer/custom')
parser.add_argument('--model_name', type=str, default='tokenizer', help='model name to be saved')
parser.add_argument('--vocab_size', type=int, default=30000)
parser.add_argument('--character_coverage', type=float, default=0.9995)
parser.add_argument('--model_type', type=str, default='bpe')
parser.add_argument('--user_defined_symbols', type=str, default='<sepra>')
args = parser.parse_args()


# 语料
if os.path.isdir(args.corpus_name_or_path):
    corpus = []
    for file in os.listdir(args.corpus_name_or_path):
        if file.endswith('.txt'):
            corpus.append(os.path.join(args.corpus_name_or_path, file))
    print(f'train {len(corpus)} corpus')
else:
    corpus = args.corpus_name_or_path
    print('train one txt')


spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=os.path.join(args.output_path, args.model_name),
    vocab_size=args.vocab_size,
    user_defined_symbols=args.user_defined_symbols.split(','),
    character_coverage=args.character_coverage,
    model_type=args.model_type,
)

