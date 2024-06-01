from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('volatility/', views.vol_tickers, name='volatility'),
    path('volatility/volcompare/', views.vol_tickers, name='compare'),
    path('volume/', views.volume_tickers, name='volume'),
    path('volume/volumecompare', views.volume_tickers, name='volumecompare'),
    path('indicators/', views.price_tickers, name='indicators'),
    path('indicators/indtemp', views.price_tickers, name='indtemp'),
]

