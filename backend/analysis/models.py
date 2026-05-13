from django.db import models
from resumes.models import Resume

class Analysis(models.Model):
    resume           = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name="analyses")
    resume_text      = models.TextField(blank=True)
    job_description  = models.TextField(blank=True)
    ats_score        = models.FloatField(default=0)
    formatting_score = models.FloatField(default=0)
    content_score    = models.FloatField(default=0)
    skills_score     = models.FloatField(default=0)
    keywords_score   = models.FloatField(default=0)
    readability_score= models.FloatField(default=0)
    found_keywords   = models.JSONField(default=list)
    missing_keywords = models.JSONField(default=list)
    strengths        = models.JSONField(default=list)
    weaknesses       = models.JSONField(default=list)
    suggestions      = models.JSONField(default=list)
    grammar_issues   = models.JSONField(default=list)
    action_verbs     = models.JSONField(default=dict)
    overall_feedback = models.TextField(blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "analyses"
        ordering = ["-created_at"]

class JobMatch(models.Model):
    resume           = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name="job_matches")
    job_title        = models.CharField(max_length=200)
    company          = models.CharField(max_length=200, blank=True)
    job_description  = models.TextField()
    match_score      = models.FloatField(default=0)
    should_apply     = models.CharField(max_length=10, default="maybe")
    matching_strengths = models.JSONField(default=list)
    gaps             = models.JSONField(default=list)
    tips             = models.JSONField(default=list)
    full_analysis    = models.TextField(blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "job_matches"
        ordering = ["-created_at"]