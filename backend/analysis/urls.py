from django.urls import path
from . import views
urlpatterns = [
    path("analyze/",    views.analyze_resume,    name="analyze-resume"),
    path("ats-match/",  views.ats_keyword_match, name="ats-keyword-match"),
    path("skill-gap/",  views.skill_gap_analysis,name="skill-gap"),
    path("history/",    views.analysis_history,  name="analysis-history"),
]