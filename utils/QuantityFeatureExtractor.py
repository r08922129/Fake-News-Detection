from .AbstractFeatureExtractor import AbstractFeatureExtractor
from nltk import word_tokenize, sent_tokenize
import numpy as np

class SentimentExtractor(AbstractFeatureExtractor):
    
    def featureName(self) -> list:

        return [
            '#Char',
            '#Words',
            '#Sentences',
            '#AvgCharsPerWord',
            '#AvgWordsPerSentence',
        ]
    def extract(self, text: str) -> np.array:

        sents = sent_tokenize(text)
        total_chars = 0
        total_words = 0
        total_sentences = len(sents)

        for sent in sents:
            words = word_tokenize(sent)
            total_words += len(words)
            for word in words:
                total_chars += len(word)
        return np.array([
            total_chars,
            total_words,
            total_sentences,
            total_chars/total_words,
            total_words/total_sentences,
        ])