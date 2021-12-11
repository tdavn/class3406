from django.db import models

# Create your models here.
class Tweet_store(models.Model):
    tweet_no = models.IntegerField()
    search_keys = models.CharField(max_length=200)
    dataframe = models.TextField()
    search_date = models.DateField(auto_now_add=True)
    p_words = models.TextField(default=' ')
    n_words = models.TextField(default=' ')

    def __str__(self):
        return self.search_keys
