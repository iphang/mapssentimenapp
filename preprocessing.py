import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk


class Preprocess:
    def __init__(self):
        pass

    def case_folding(self, text):
        text = text.lower()
        return text

    def cleansing(self, text):
       # remove URL first
        text = re.sub(r"http\S+", '', text)  # remove link
        # Remove every URL
        text = re.sub(
            '((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
        # remove mention and hashtag
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)  # remove hashtag
        # remove other unnecessary characters
        text = re.sub('RT|rt', '', text)  # remove retweet symbol
        # remove text inside square brackets
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(
            """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
        text = re.sub('\n', ' ', text)  # Remove every '\n'
        text = re.sub('\s+', ' ', text).strip()  # Remove extra spaces and trim
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = text.encode('ascii', 'ignore').decode(
            'ascii')  # Remove non-ASCII characters
        text = re.sub('[^0-9a-zA-Z]+', ' ', text)
        return text

    # normalization
    def normalization(self, text):
        alay_dict = pd.read_csv('data/new_kamusalay.csv',
                                encoding='latin-1', header=None)
        alay_dict_map = dict(zip(alay_dict[0], alay_dict[1]))
        text = ' '.join(
            [alay_dict_map[word] if word in alay_dict_map else word for word in text.split()])
        return text

    def filtering(self, text):
        stopwords = pd.read_csv('data/stopword.csv',
                                encoding='latin-1')
        stopwords = stopwords.rename(columns={0: 'stopword'})
        text = ' '.join(
            ['' if word in stopwords.stopword.values else word for word in text.split(' ')])
        text = re.sub('  +', ' ', text)  # Remove extra spaces
        text = text.strip()
        return text

    def stemming(self, text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = stemmer.stem(text)
        return text

    def tokenize(self, text):
        text = nltk.tokenize.word_tokenize(text)
        return text
