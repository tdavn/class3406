import tweepy as tw
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
import re
import string

from tensorflow.keras.preprocessing.sequence import pad_sequences

# from tensorflow.keras.preprocessing.text import Tokenizer

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Tweet collector
def tweet_collector(keys, lang):
    ACCESS_TOKEN ="917808967274741762-ZV1vZvXxRRZRGL8mVZkIaR465oDdSG6"
    ACCESS_TOKEN_SECRET ="noTjc6A8Vw4s2oo9UUUHdhyfPkfeGAe8PiUYfuEv8TelS"
    CONSUMER_API_KEY="Srkf7PXrUh5dqgbk228Q0ijg0"
    CONSUMER_API_SECRET="sij33zHlmljyjVM8jf0xGrKneVFo5MQMDDUfCYAomCiwByUlHl"
    # Create and authentication object
    auth = tw.OAuthHandler(CONSUMER_API_KEY, CONSUMER_API_SECRET)
    # Setting your access token and secret
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    # Create the API object passing the auth object
    api = tw.API(auth)
    tweets = tw.Cursor(api.search_tweets, q=keys, lang=lang).items(50)
    # Store tweets in a list
    tweet_component = []
    for tweet in tweets:
        tweet_component.append([tweet.user.id, tweet.created_at,
                        tweet.user.screen_name, tweet.user.location, tweet.text.encode('utf-8')])
    # Convert to DataFrame that contains information including username, location.
    df_tweet = pd.DataFrame(tweet_component, columns=['ID', 'Time', 'User name', 'Location', 'Text'])

    return df_tweet


def preProcess_data(tweet):
    stop_words = stopwords.words('english')
    # lower text
    tweet = tweet.lower()
    # remove urls
    tweet = re.sub(r'http\S+', ' ', str(tweet))
    # remove html tags
    tweet = re.sub(r'<.*?>',' ', tweet)
    # remove tweets containing digits
    tweet = re.sub('\w*\d\w*', ' ', tweet)
    # remove hashtags
    tweet = re.sub(r'#\w+',' ', tweet)
    # remove mentions
    tweet = re.sub(r'@\w+',' ', tweet)
    # remove anything not a letter
    tweet = re.sub(r'\W+', ' ', tweet)
    tweet = re.sub(r'rt|RT|retweet', '', tweet)
    tweet = tweet.strip()
    #removing words with length less than 2 or in stop words
    tweet = tweet.split()
    tweet = [w for w in tweet if len(w)>1]
    tweet = " ".join([word for word in tweet if not word in stop_words])
    return tweet


def cleaned_df(df):
    '''return a padded tweed data frame'''
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(df['Text'].to_numpy())


    # def my_pipeline(text):
    #     text_new = preProcess_data(text)
    #     X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    #     X = pad_sequences(X, maxlen=20, padding='post')
    #     return X

    df['cleaned'] =  df['Text'].apply(preProcess_data)


    # def sen_eval(cln_tweet):
    #     predictions = model.predict(cln_tweet)
    #     sentiment = int(np.argmax(predictions))
    #     probability = max(predictions.tolist()[0])
    #     if sentiment == 0:
    #          return 'Negative'
    #     elif sentiment == 1:
    #          return 'Neutral'
    #     elif sentiment == 2:
    #          return 'Postive'

    # df['Sent'] =  df['padded'].apply(sen_eval)

    return df
