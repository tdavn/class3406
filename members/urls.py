from django.urls import path
from .views import UserRegisterView
urlpatterns = [
    path('members/', UserRegisterView.as_view(), name='register'),
]
