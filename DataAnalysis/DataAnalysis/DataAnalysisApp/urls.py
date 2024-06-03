from django.urls import path
from django.conf.urls.static import static
from DataAnalysisApp import views
from DataAnalysis import settings


app_name = 'DataAnalysisApp'
urlpatterns = [
    path('', views.index, name='index'),
    path('logout/', views.logout, name='logout'),
    path('sign-in/', views.signIn, name='signIn'),
    path('sign-up/', views.signUp, name='signUp'),
    path('sentence/', views.sentence, name='sentence'),
    path('opinion-classification/', views.opinionClassification, name='opinionClassification'),
    path('opinion-analysis/', views.opinionAnalysis, name='opinionAnalysis'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)