from nltk.tokenize.casual import TweetTokenizer
import pickle

tokenizer = TweetTokenizer()

ayo = 1
karan = 0
grammer = ''
karan_answers = []
ayo_questions = []

# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = s.lower().strip()
    return s
 # Read the file and split into lines
lines = open('/home/karan/Downloads/ayo.txt', encoding='utf-8').\
    read().strip().split('\n')


for line in lines:

    try:
        k = normalizeString(line.split('Karanbir Singh Chahal:')[1])
        grammer += ' ' + k
        if karan == 0:
            tmp = k
            karan_answers.append(tmp)
            karan = 1
            ayo = 0
        else:
            tmp = karan_answers[-1]
            tmp = tmp + ' ' + k 
            karan_answers[-1] = tmp

    except Exception as e:
        pass

    try:   
        a = normalizeString(line.split('Ayotakshee:')[1])
        grammer += ' ' + a
        if ayo == 0:
            tmp = a
            ayo_questions.append(tmp)
            ayo = 1
            karan = 0
        else:
            tmp = ayo_questions[-1]
            tmp = tmp + ' ' + a 
            ayo_questions[-1] = tmp

    except Exception as e:
        pass



class Vocab:
    
    def __init__(self):
        self.id2word = []
        self.word2id = {}

        self.id2word.append('<SOS>')
        self.id2word.append('<EOS>')
        self.word2id['<SOS>'] = 0
        self.word2id['<EOS>'] = 1

        self.id = 2
        self.examples = 0

    def add(self,word):
        if word not in self.word2id:
            self.word2id[word] = self.id
            self.id2word.append(word)
            self.id +=1
    
    def word(self,id):
        return self.id2word[id]
    
    def idx(self,word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id['<EOS>']
    
    def length(self):
        return self.id

tokens = tokenizer.tokenize(grammer)

vocab = Vocab()
for t in tokens:
    vocab.add(t)

main_k_answers = []

for i in karan_answers:
    i = tokenizer.tokenize(i)
    k_tokens = []
    for j in i:
        k_tokens.append(vocab.idx(j))
    main_k_answers.append(k_tokens)


main_a_questions = []

for i in ayo_questions:
    i = tokenizer.tokenize(i)
    a_tokens = []
    for j in i:
        a_tokens.append(vocab.idx(j))
    main_a_questions.append(a_tokens)

print(len(main_a_questions))
print(len(main_k_answers))
