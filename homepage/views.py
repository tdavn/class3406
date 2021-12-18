from typing import Text
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.http import HttpResponse, HttpResponseRedirect
from .models import Tweet_store
from rest_framework import viewsets
from .serializers import SentimentSerializer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
import io
import urllib, base64
from wordcloud import WordCloud

from pathlib import Path

# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.models import load_model


# import tweepy as tw
# import pandas as pd
from . import functions
# loaded_model = tf.keras.models.load_model('sentiment.h5')

sid_obj = SentimentIntensityAnalyzer()
BASE_DIR = Path(__file__).resolve().parent



# Create your views here.
# class HomepageView(TemplateView):
#     '''Display homepage'''
#     template_name = 'homepage/index.html'

class SentimentView(viewsets.ModelViewSet):
    queryset = Tweet_store.objects.all()
    serializer_class = SentimentSerializer


def receiver(request):
    if request.method == 'POST':
        try:
            date = request.POST['date']
            country = request.POST['country']
            language = request.POST['lang']
            query = request.POST['message']
            # process the query
            tmp = query.split(',')
            keywords = ' OR'.join(x for x in tmp)

            print(keywords)
            # Tweet collector
            df_tweet = functions.tweet_collector(keywords, language)
            print(df_tweet.head(3)[['Text']])

            cleaned_df = functions.cleaned_df(df_tweet)
            tweet_no = len(cleaned_df)
            print('final tweet:', tweet_no)
            print(cleaned_df.head()[['cleaned']])

            # Vader
            def sentiment_scores(sentence):
                sentiment_dict = sid_obj.polarity_scores(sentence)
                if sentiment_dict['compound'] >= 0.05 :
                    return "Positive"
                elif sentiment_dict['compound'] <= - 0.05 :
                    return "Negative"
                else :
                    return "Neutral"

            cleaned_df['sent'] = cleaned_df['cleaned'].apply(sentiment_scores)
            print(cleaned_df.head()[['cleaned', 'sent']])

            df_html = cleaned_df.head(5)[['Text', 'cleaned', 'sent']].to_html()

            p_words = ''
            for i in cleaned_df[cleaned_df['sent']=='Positive']['cleaned']:
                p_words = p_words + ' ' + i

            n_words = ''
            for i in cleaned_df[cleaned_df['sent']=='Negative']['cleaned']:
                n_words = n_words + ' ' + i

            fig = cleaned_df['sent'].value_counts().plot(kind='bar', rot=20).get_figure()
            fig.savefig(BASE_DIR / "static/homepage/img/sent.png")
            Tweet_store.objects.create(tweet_no = tweet_no, search_keys=query, dataframe=df_html, p_words=p_words, n_words=n_words)

            # Loading models
            # from tensorflow.keras.preprocessing.text import Tokenizer
            # tokenizer = Tokenizer()
            # tokenizer.fit_on_texts(cleaned_df['Text'].to_numpy())

            # def padding(text):
            #     X = tokenizer.texts_to_sequences(pd.Series(text).values)
            #     X = pad_sequences(X, maxlen=20, padding='post')
            #     return X

            # adding padded columns
            # cleaned_df['padded'] = cleaned_df['cleaned'].apply(padding)
            # print(padded_df.head(3))

            # predictions
            # def predicting(text):
            #     predictions = loaded_model.predict(text)
            #     sentiment = int(np.argmax(predictions))
            #     probability = max(predictions.tolist()[0])
            #     if sentiment == 0:
            #          return 'Negative'
            #     elif sentiment == 1:
            #          return 'Neutral'
            #     elif sentiment == 2:
            #          return 'Postive'
            #
            # cleaned_df['predicted'] = cleaned_df['padded'].apply(predicting)

            # df_html = cleaned_df.head(5)[['Text', 'cleaned']].to_html()
            # Tweet_store.objects.create(tweet_no = tweet_no, search_keys=query, dataframe=df_html)
            # print(df_html)
            # Produce a data frame contain sentiment columns

            # final_df = functions.sent_df(cleaned_df)
            # print(cleaned_df.head(3)[['padded', 'predicted']])
            return redirect('success/')
        except Exception as e:
            print(e)


    return render(request, 'homepage/index.html' )

def word_cloud(text, name):
    wc = WordCloud(max_font_size=50, max_words=100, background_color = 'white')
    wc= wc.generate(text)
    wc.to_file(f"{BASE_DIR}/static/homepage/img/wordcloud_{name}.png")
    plt.figure(figsize= (10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)  # rewind the data
    string = base64.b64encode(image.read())

    image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
    return image_64

# def success(request, *args, **kwargs):
#     '''Using GET to listen to endpoints'''
#     import json
#     import requests
#
#     api_request = requests.get('/apiclassifier/')
#     api = json.loads(api_request.content)
#     p_words = api['p_words']
#     return render(request, 'homepage/board.html', {'api':api})

def success(request, *args, **kwargs):
    '''Django direct access to db'''
    tweet_dt = Tweet_store.objects.last()
    # tweet_dt = Tweet_store.objects.filter()
    df = tweet_dt.dataframe
    search_keys = tweet_dt.search_keys
    tweet_no = tweet_dt.tweet_no
    p_words = tweet_dt.p_words
    n_words = tweet_dt.n_words
    import os
    SETTINGS_DIR = os.path.dirname(__file__)
    print(SETTINGS_DIR)
    print(BASE_DIR)
    # print('negative: ', n_words)
    # wordcloud_p = word_cloud(p_words)
    wordcloud_n = word_cloud(n_words, 'n')
    wordcloud_p = word_cloud(p_words, 'p')

    return render(request, 'homepage/board.html', {'search_keys': search_keys, 'tweet_no':tweet_no, 'df':df, 'wordcloud_n': wordcloud_n, 'wordcloud_p':wordcloud_p})
