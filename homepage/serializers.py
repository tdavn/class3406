from rest_framework import serializers
from .models import Tweet_store

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tweet_store
        fields = ('id','tweet_no', 'search_date', 'search_keys', 'dataframe')
