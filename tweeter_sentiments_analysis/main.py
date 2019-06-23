import configparser
import re
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import tweepy
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud

analyser = SentimentIntensityAnalyzer()
parser = configparser.ConfigParser()


class TwitterSentimentsAnalysis(object):
    def __init__(self, configdict={}):
        self._configdict = configdict
        self._twitter = None

    @property
    def twitter(self):
        if not self._twitter:
            auth = tweepy.OAuthHandler(
                self._configdict["consumer_key"], self._configdict["consumer_secret"]
            )
            auth.set_access_token(
                self._configdict["oauth_token"], self._configdict["oauth_secret"]
            )
            # creating the API object
            self._twitter = tweepy.API(auth)
        return self._twitter

    @staticmethod
    def _remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, "", input_txt)
        return " ".join(input_txt.split())

    def _clean_tweets(self, lst):
        # remove twitter Return handles (RT @xxx:)
        lst = np.vectorize(self._remove_pattern)(lst, "RT @[\w]*:")
        # remove twitter handles (@xxx)
        lst = np.vectorize(self._remove_pattern)(lst, "@[\w]*")
        # remove URL links (httpxxx)
        lst = np.vectorize(self._remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
        # remove special characters, numbers, punctuations (except for #)
        lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
        return lst

    def search_tweets(self, search, count=2000):
        results = []
        for tweet in tweepy.Cursor(self.twitter.search, q=search, lang="en").items(count):
            results.append(tweet)
        if results:
            list_of_ids = [result.id for result in results]
            data_set = pd.DataFrame(list_of_ids, columns=["id"])
            data_set["created_at"] = [result.created_at for result in results]
            try:
                data_set["hashtag"] = list(
                    set(
                        hashtags.get("text", None)
                        for result in results
                        if result.entities.get("hashtags")
                        for hashtags in result.entities.get("hashtags", None)
                    )
                )
            except Exception:
                data_set["hashtag"] = [
                    result.entities.get("hashtags") for result in results
                ]

            data_set["retweet_count"] = [result.retweet_count for result in results]
            data_set["text"] = [result.text for result in results]
            data_set["user_followers"] = [
                result.user.followers_count for result in results
            ]
            data_set["user_name"] = [result.author.screen_name for result in results]
            data_set["user_location"] = [result.user.location for result in results]
            # Clean and Remove duplicates
            cleaned_texts = self._clean_tweets(data_set["text"])
            cleaned_texts = list(cleaned_texts)
            for index, text in enumerate(cleaned_texts):
                data_set.at[index, "text_duplicates"] = text

            data_set.drop_duplicates("text_duplicates", inplace=True)
            data_set.reset_index(drop=True, inplace=True)
            data_set.drop("text", axis=1, inplace=True)
            data_set.rename(columns={"text_duplicates": "text"}, inplace=True)
            return data_set

    @staticmethod
    def _sentiment_analyzer_scores(text):
        score = analyser.polarity_scores(text)
        lb = score["compound"]
        if lb >= 0.05:
            return "Positive"
        elif (lb > -0.05) and (lb < 0.05):
            return "Neutral"
        else:
            return "Negative"

    def generate_sentiments(self, data_set):
        assert isinstance(data_set, pd.core.frame.DataFrame)
        texts = data_set.get("text")
        for text in texts:
            sentiment = self._sentiment_analyzer_scores(text)
            data_set.at[index, "SentimentalPolarityVader"] = sentiment
        return data_set

    def generate_wordcloud(self, words, image_title="Sentiment"):
        wordcloud = WordCloud(
            background_color="black",
            stopwords=STOPWORDS,
            width=1600,
            height=800,
            random_state=1,
            colormap="jet",
            max_words=50,
            max_font_size=200,
        ).generate(words)

        plt.title(image_title, fontsize=20, color="Red")
        plt.figure()
        plt.axis("off")
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.savefig(f"{image_title}.png", bbox_inches="tight", dpi=300)

    @staticmethod
    def save_to_csv(data_set, filename="data_frame.csv"):
        assert isinstance(data_set, pd.core.frame.DataFrame)
        data_set.to_csv(file_name, sep="\t", encoding="utf-8")


if __name__ == "__main__":
    config_file = Path("config.ini")
    if config_file:
        parser.read(config_file)
        configdict = {
            section: dict(parser.items(section)) for section in parser.sections()
        }

        twitterAPI = TwitterSentimentsAnalysis(configdict=configdict["Twitter Keys"])

        results = twitterAPI.search_tweets("food")
        results = twitterAPI.generate_sentiments(results)
        words = " ".join(results["text"])
        results = twitterAPI.generate_wordcloud(words)
        twitterAPI.save_to_csv(results)
