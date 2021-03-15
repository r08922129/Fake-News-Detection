from .AbstractFeatureExtractor import AbstractFeatureExtractor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

nltk.download('vader_lexicon')
class SentimentExtractor(AbstractFeatureExtractor):
    
    def featureName(self) -> list:

        return [
            'SentimentNEG',
            'SentimentNEU',
            'SentimentPOS',
            'SentimentCompund',
        ]
    def extract(self, text: str) -> np.array:

        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(text)
        return np.array([
          score['neg'],
          score['neu'],
          score['pos'],
          score['compound'],
        ])