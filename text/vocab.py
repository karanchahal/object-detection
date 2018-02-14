import gensim
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
import coloredlogs, logging

# Create a logger object.
logger = logging.getLogger(__name__)
PROJECT_DIR = '/home/karan/attention-caption/'

class Vocab:
    
    def __init__(self):
        self.id2word = []
        self.word2id = {}

        self.id2word.append('<SOS>')
        self.id2word.append('<EOS>')
        self.id2word.append('<UNK>')
        self.word2id['<SOS>'] = 0
        self.word2id['<EOS>'] = 1
        self.word2id['<UNK>'] = 2

        self.id = 3
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
            return self.word2id['<UNK>']
    
    def length():
        return self.id

class WordModel:
    
    def __init__(self):
        self.word_model = None
        self.word_err_file = None
        self.word_errors = {}
        self.vocab = Vocab()

    def load_model(self):
        logger.warning("Loading word model")
        self.word_model = gensim.models.KeyedVectors.load_word2vec_format('/home/karan/embeddings/lex.vectors', binary=False)

    def write_to_log(self,sentence,filename):
        logger.warning(str(sentence) + ' writing to log')
        self.word_err_file = open(filename, "a+")
        self.word_err_file.write(sentence + '\n')

    def close_log(self):
        self.word_err_file.close()
        self.word_err_file = None

    def collect_errors(self,word,filename):
        if word not in self.word_errors:
            self.word_errors[word] = 1
            self.write_to_log(word,filename)

    def build_vocab(self,captions):
        logger.warning("Building vocabulary")
        remove = 0

        for c in range(len(captions[0])):
            for d in range(len(captions)):

                caption = captions[d][c]

                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(caption)

                for word in tokens:
                    word = word.lower()
                    if word not in self.word_model:
                        remove = 1
                        logger.error(str(word) + ' does not exist')
                        self.collect_errors(word,filename=PROJECT_DIR + '/errors/word_error.log')
                        break
                    else:
                        self.vocab.add(word)

                if remove == 1:
                    break
            if remove == 1:
                remove = 0
            else:
                self.vocab.examples += 1
                    
    def parse(self,captions):
        parsed_captions = []
        for c in range(len(captions[0])):
            example_captions = []
            for d in range(len(captions)):
                caption = captions[d][c]
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(caption)
                ids = []
                for word in tokens:
                    word = word.lower()
                    ids.append(self.vocab.idx(word))
                ids.append(self.vocab.idx('<EOS>'))
                
                example_captions.append(ids)   
            parsed_captions.append(example_captions)
        
        return parsed_captions

    def save(self,filename):
        pickle.dump(self.vocab,open(filename,"wb"))
    
    def load(self,filename):
        self.vocab = pickle.load(open(filename,"rb"))