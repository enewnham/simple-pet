from functools import lru_cache
from pathlib import Path
import re

import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .user_text import user_prose


NEGATIVE_THRESHOLD = -0.45
REACTION_CHANCE = 0.5
COOLDOWN_SECONDS = 90
DUPLICATE_SECONDS = 2


@lru_cache(maxsize=1)
def analyzer():
    directory = str(Path(__file__).with_name("data").resolve())
    nltk.data.path.insert(0, directory)
    return SentimentIntensityAnalyzer(lexicon_file="vader_lexicon.txt")


def sentiment_score(text):
    sentences = re.split(r"(?<=[.!?])\s+|\n+", user_prose(text))
    return min((analyzer().polarity_scores(sentence)["compound"] for sentence in sentences if sentence.strip()), default=0.0)
