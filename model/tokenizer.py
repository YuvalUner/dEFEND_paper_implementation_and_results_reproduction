from spacy.lang.en import English


class Tokenizer:

    def __init__(self):
        """
        Initialize the Tokenizer object with the spacy model.
        """
        nlp = English()
        self.tokenizer = nlp.tokenizer


    def tokenize(self, text):
        return [token.text for token in self.tokenizer(text)]

    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)