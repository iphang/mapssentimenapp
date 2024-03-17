import nltk


class CustomTokenizer:
    def __init__(self):
        pass

    def word_tokenize_wrapper(self, text):
        text = nltk.tokenize.word_tokenize(text)
        return text
