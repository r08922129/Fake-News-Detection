from src.utils.AbstractFeatureExtractor import AbstractFeatureExtractor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

class SentimentExtractor(AbstractFeatureExtractor):
    
    def __init__(self, config):
        pass

    def featureName(self) -> list:

        return [
            'SentimentNEG',
            'SentimentNEU',
            'SentimentPOS',
            'SentimentCompund',
        ]
    def extract(self, data : list) -> np.array:

        sid = SentimentIntensityAnalyzer()

        out = []
        for text in data:
            score = sid.polarity_scores(text)
            out.append(
                np.array([
                    score['neg'],
                    score['neu'],
                    score['pos'],
                    score['compound'],
                ])
            )
        return np.array(out)