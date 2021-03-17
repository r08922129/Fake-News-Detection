from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import nltk
import os
import numpy as np
import stanza

class BOWExtractor(AbstractFeatureExtractor):

    def __init__(self, config):

        self.vocabSize = config.vocabSize
        self.words = {}
        self.setOfNER = set()
        self._countBOWFromCorpus(config.pathesToFiles)

    def featureName(self) -> list:
        return [keyValue[0] for keyValue in sorted(self.words.items(), key=lambda x : x[1])]

    def extract(self, data : list) -> np.array:

        out = []
        for text in data:
            words = self.tokenizer(text)
            array = np.zeros(len(self.words))
            numberOfWords = 0
            for word in words:
                if word in self.words:
                    index = self.words[word]
                    array[index] += 1
                    numberOfWords += 1
            if numberOfWords:
                out.append(array/numberOfWords)
            else:
                out.append(np.zeros(len(self.words)))

        return np.array(out)

    def _countBOWFromCorpus(self, pathesToFiles):
        corpus = []
        for filePath in pathesToFiles:
            with open(filePath) as f:
                corpus.append(f.read())

        # count NER
        nlp = stanza.Pipeline(lang='en', processors='tokenize, ner')
        for doc in corpus:
            doc = nlp(doc)
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    if token.ner is not "O":
                        self.setOfNER.add(token.text)

        vectorizer = TfidfVectorizer(stop_words='english', tokenizer=self.tokenizer)
        X = vectorizer.fit_transform(corpus)
        words = vectorizer.get_feature_names()
        # words ordered by count.
        words = np.array(words, dtype=str)[vectorizer.idf_.argsort()][::-1]
        index = 0
        count = self.vocabSize
        for word in words:
            if count:
                if word not in self.setOfNER:
                    self.words[word] = index
                    index += 1
                    count -= 1
            else:
                break

    def _get_wordnet_pos(self, tag):

        if tag.startswith('J'):
            return wordnet.ADJ

        elif tag.startswith('V'):
            return wordnet.VERB

        elif tag.startswith('N'):
            return wordnet.NOUN
        
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def tokenizer(self, text):

        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        taggedWords = nltk.pos_tag(words)
        out = []
        for wordAndTag in taggedWords:
            if wordAndTag[0] in self.setOfNER:
                out.append(wordAndTag[0])
            else:
                wordnet_pos = self._get_wordnet_pos(wordAndTag[1]) or wordnet.NOUN
                out.append(lemmatizer.lemmatize(wordAndTag[0], pos=wordnet_pos))
        return out