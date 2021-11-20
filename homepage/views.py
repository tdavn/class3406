from typing import Text
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.http import HttpResponse, HttpResponseRedirect
from .models import Tweet_store
from rest_framework import viewsets
from .serializers import SentimentSerializer

# import tensorflow as tf 
# from tensorflow.keras.preprocessing.text import Tokenizer
# from keras.models import load_model


# import tweepy as tw
# import pandas as pd
from . import functions
# loaded_model = tf.keras.models.load_model('sentiment.h5')



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

            
            # Tweet collector
            df_tweet = functions.tweet_collector(keywords, language)
            cleaned_df = functions.cleaned_df(df_tweet)
            tweet_no = len(cleaned_df)
            df_html = cleaned_df.head(5)[['Text', 'cleaned']].to_html()
            Tweet_store.objects.create(tweet_no = tweet_no, search_keys=query, dataframe=df_html)
            # print(df_html)
            # Produce a data frame contain sentiment columns
            
            # final_df = functions.sent_df(df)
            return redirect('success/')
        except Exception as e:
            print(e)


    return render(request, 'homepage/index.html' )


def success(request, *args, **kwargs):
    import json
    import requests
    
    api_request = requests.get('http://localhost:8000/apiclassifier/')
    try:
        api = json.loads(api_request.content)
        
    except Exception as err:
        api = 'Error...'
    return render(request, 'homepage/board.html', {'api':api})

        
        



        # ouput is df

        # Processing step
        

        # Model load and applying
        



    # def post(self, request):
    #     info = self.request.POST
    #     ContactMessage.objects.create(
    #     name = info['name'],
    #     email = info['email'],
    #     message = info['message']
    #     )
    #     context = {
    #     'temp_mes': 'Your message has been recored. Thank you!',
    #     }
    #     return render(request, self.template_name, context)