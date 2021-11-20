from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()
router.register('classifier', views.SentimentView)

urlpatterns = [
    path('', views.receiver, name = 'home'),
    path('api', include(router.urls)),
    path('success/', views.success, name='success'),
] 