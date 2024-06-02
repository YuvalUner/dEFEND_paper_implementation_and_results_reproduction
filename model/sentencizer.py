import spacy

class Sentencizer:

    def __init__(self):
        """
        Initialize the Sentencizer object with the spacy model.
        If the model is not already downloaded, download it using the command:

        >>> python -m spacy download en_core_web_sm

        """
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('sentencizer')

    def sentencize(self, text):
        return [sent.text for sent in self.nlp(text).sents]

    def __call__(self, *args, **kwargs):
        return self.sentencize(*args, **kwargs)