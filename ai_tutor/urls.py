from django.urls import path

from ai_tutor import views

urlpatterns = [
    path("",  views.home_page, name='home_page'),
    path("api/ask/question/", views.AskQuestionView.as_view(), name='ask-question')
]