from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
import numpy as np
from nltk.tokenize import SyllableTokenizer
from nltk import word_tokenize, sent_tokenize
import re

class ReadabilityExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):
        pass
    
    def featureName(self) -> list:
        
        return [
          'FleschReadingEaseIndex',
         'FleschKincaidGradeLevel',
          'AutomatedReadabilityIndex',
          'GunningFoxIndex',
          'ColemanLiauIndex',
          '#Words',
          '#Syllables',
          '#Polysyllables',
          '#Chars',
          '#ComplexWords'
          ]
    
    def extract(self, data : list) -> np.array:

        SSP = SyllableTokenizer()
        suffix_pattern = re.compile("ed$|es$|ed$|ing$")
        out = []
        for text in data:
            sents = sent_tokenize(text)
            total_words = 0
            total_sentences = len(sents)
            total_syllables = 0
            total_polysyllables = 0
            total_chars = 0
            total_complex_word = 0
            
            for sent in sents:
                
                words = word_tokenize(sent)
                total_words += len(words)
                
                for word in words:

                    total_chars += len(word)
                    syllables = SSP.tokenize(word) 
                    total_syllables += len(SSP.tokenize(word))
                    if len(syllables) > 2:
                        total_polysyllables += 1
                    if len(SSP.tokenize(re.sub(suffix_pattern, '', word))) > 2:
                        total_complex_word += 1
            
            array = np.array([
                self.valueOfFleschReadingEaseIndex(total_words, total_sentences, total_syllables),
                self.valueOfFleschKincaidGradeLevel(total_words, total_sentences, total_syllables),
                self.valueOfAutomatedReadabilityIndex(total_chars, total_words, total_sentences),
                self.valueOfGunningFoxIndex(total_words, total_sentences, total_complex_word),
                self.valueOfColemanLiauIndex(total_chars, total_words, total_sentences),
                total_words,
                total_syllables,
                total_polysyllables,
                total_chars,
                total_complex_word,
                            ])
            out.append(array)

        return np.array(out)
    
    def valueOfFleschReadingEaseIndex(self, total_words, total_sentences, total_syllables):
        ''' Value of Flesch Reading Ease Index
        Reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        '''
        return 206.835 - 1.015 * (total_words/total_sentences) - 84.6 * (total_syllables/total_words)
    
    def valueOfFleschKincaidGradeLevel(self, total_words, total_sentences, total_syllables):
        ''' Value of Flesch Kincaid Grade Level
        Reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        '''
        return 0.39 * (total_words/total_sentences) + 11.8 * (total_syllables/total_words)

    def valueOfAutomatedReadabilityIndex(self, total_chars, total_words, total_sentences):
        ''' Value of Automated Readability Index
        Reference: https://en.wikipedia.org/wiki/Automated_readability_index
        '''
        return 4.71 * (total_chars/total_words) + 0.5 * (total_words/total_sentences)
    
    def valueOfGunningFoxIndex(self, total_words, total_sentences, total_complex_word):
        ''' Value of Gunning Fox Index
        Reference: https://en.wikipedia.org/wiki/Gunning_fog_index
        '''
        return 0.4 * ((total_words/total_sentences) + 100 * (total_complex_word/total_words))

    def valueOfColemanLiauIndex(self, total_chars, total_words, total_sentences):
        ''' Value of Coleman-Liau Index
        Reference: https://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        '''
        L = 100 * (total_chars/total_words)
        S = 100 * (total_sentences/total_words)
        return 0.0588 * L - 0.296 * S - 15.8 