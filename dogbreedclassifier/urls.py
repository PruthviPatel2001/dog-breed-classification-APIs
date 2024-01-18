from django.urls import path
from .views import predict_dog_breed,welcome

urlpatterns = [
    path('predict/', predict_dog_breed, name='predict_dog_breed'),
    path('welcome/', welcome, name='welcome'),
]