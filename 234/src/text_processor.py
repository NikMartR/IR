import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords', quiet=True)

class TextProcessor:
    def __init__(self, language: str = 'russian'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    def process(self, text: str) -> list[str]:
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()

        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)

        return processed_tokens