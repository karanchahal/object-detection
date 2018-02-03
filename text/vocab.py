import gensim
import nltk
from nltk.tokenize import RegexpTokenizer
import coloredlogs, logging

# Create a logger object.
logger = logging.getLogger(__name__)
PROJECT_DIR = '/home/karan/attention-caption/'
word_model = None
word_err_file = None
word_errors = {}


def load_model():
    logger.warning("Loading word model")
    global word_model
    word_model = gensim.models.KeyedVectors.load_word2vec_format('/home/karan/embeddings/lex.vectors', binary=False)


def write_to_log(sentence,filename):
    global word_err_file
    logger.warning(str(sentence) + ' writing to log')
    word_err_file = open(filename, "a+")
    word_err_file.write(sentence + '\n')

def close_log():
    global word_err_file
    word_err_file.close()
    word_err_file = None

def collect_errors(word,filename):
    global word_errors
    if word not in word_errors:
        word_errors[word] = 1
        write_to_log(word,filename)

def create(captions):
    logger.warning("Creating vocabulary")
    for batch in captions:
        for caption in batch:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(caption)
            for word in tokens:
                word = word.lower()
                if word not in word_model:
                    logger.error(str(word) + ' does not exist')
                    collect_errors(word,filename=PROJECT_DIR + '/errors/word_error.log')