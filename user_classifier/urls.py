from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home.as_view(), name='learner_home'),
    path('classification', views.Classification.as_view(), name='learner_classification'),
    path('regression', views.Regression.as_view(), name='learner_regression'),
    path('clustering', views.Clustering.as_view(), name='learner_clustering')
]
