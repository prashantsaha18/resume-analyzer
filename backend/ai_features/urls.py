from django.urls import path
from . import views
urlpatterns = [
    path("improve-resume/",    views.improve_resume,        name="improve-resume"),
    path("enhance-bullet/",    views.enhance_bullet,        name="enhance-bullet"),
    path("detect-weaknesses/", views.detect_weaknesses,     name="detect-weaknesses"),
    path("check-grammar/",     views.check_grammar,         name="check-grammar"),
    path("cover-letter/",      views.generate_cover_letter, name="cover-letter"),
    path("job-roles/",         views.recommend_job_roles,   name="job-roles"),
    path("job-apply/",         views.job_apply_assistant,   name="job-apply"),
    path("fake-detect/",       views.detect_fake_resume,    name="fake-detect"),
    path("portfolio/",         views.generate_portfolio,    name="portfolio"),
]