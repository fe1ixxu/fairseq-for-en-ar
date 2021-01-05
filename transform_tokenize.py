from transformers import AutoTokenizer, AutoModel
from sacremoses import MosesTokenizer, MosesDetokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
fo = open("/export/c12/haoranxu/fairseq/wmt14en-de/debpe/train.de", encoding="utf-8")
fw = open("/export/c12/haoranxu/fairseq/wmt14en-de/bertdata/train.de", "w", encoding="utf-8")
detok = MosesDetokenizer("de")
line = fo.readline()
while(line):
    line = detok.detokenize(line.split()) 
    toks = tokenizer.tokenize(line)
    toks = " ".join(toks)
    fw.writelines([toks, "\n"])
    line = fo.readline()

fo.close()
fw.close()


# fo = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_test/vocab.txt", encoding="utf-8")
# fw = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_test/vocab2.txt", "w", encoding="utf-8")
# lines = fo.readlines()
# num = len(lines)
# for line in lines:
#     line = line.strip()
#     fw.writelines([line, " ", str(num), "\n"])
#     num -= 1

# fo.close()
# fw.close()