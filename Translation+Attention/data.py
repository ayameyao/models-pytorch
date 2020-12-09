import unicodedata
import string
import re
import random

SOS_token = 0
EOS_token = 0


class Lang:
	def __init__(self,name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1


# Unicode
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		)	

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r"\1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def readLangs(lang1, lang2, reverse=False):
    print("------------Reading lines... ----------------")

    # 读取行，并分离
    lines = open('./data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


#句子最大长度为10个，包括标点符号
MAX_LENGTH = 10

eng_prefixes = (
	"i am ", "i m ",
	"he is", "he s ",
	"she is", "she s ",
	"you are", "you re ",
	"we are", "we re ",
	"they are", "they re "
)

def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and \
		len(p[1].split(' ')) < MAX_LENGTH and \
		p[1].startswith(eng_prefixes)

def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]


'''
准备数据过程：
（1）读取文本文件并拆分为行，将行拆分为成对
（2）规范文本，按长度和内容过滤
（3）成对建立句子中的单词列表
'''
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))
