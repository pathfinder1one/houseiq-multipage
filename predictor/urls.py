from django.urls import path
from . import views

urlpatterns = [
    path('',           views.home,          name='home'),
    path('predict/',   views.predict_page,  name='predict'),
    path('analytics/', views.analytics_page,name='analytics'),
    path('about/',     views.about_page,    name='about'),
    path('api/predict',views.api_predict,   name='api_predict'),
    path('api/metrics',views.api_metrics,   name='api_metrics'),
]
