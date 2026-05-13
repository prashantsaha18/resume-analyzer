from django.db import models
from users.models import User

class Resume(models.Model):
    TEMPLATE_CHOICES = [("fresher","Fresher"),("developer","Developer"),("internship","Internship"),("custom","Custom")]
    user          = models.ForeignKey(User, on_delete=models.CASCADE, related_name="resumes")
    name          = models.CharField(max_length=200, default="My Resume")
    template      = models.CharField(max_length=20, choices=TEMPLATE_CHOICES, default="custom")
    version_number= models.PositiveIntegerField(default=1)
    is_primary    = models.BooleanField(default=False)
    full_name     = models.CharField(max_length=200, blank=True)
    email         = models.EmailField(blank=True)
    phone         = models.CharField(max_length=30, blank=True)
    location      = models.CharField(max_length=200, blank=True)
    linkedin_url  = models.URLField(blank=True)
    github_url    = models.URLField(blank=True)
    portfolio_url = models.URLField(blank=True)
    summary       = models.TextField(blank=True)
    education         = models.JSONField(default=list, blank=True)
    experience        = models.JSONField(default=list, blank=True)
    projects          = models.JSONField(default=list, blank=True)
    certifications    = models.JSONField(default=list, blank=True)
    technical_skills  = models.JSONField(default=list, blank=True)
    soft_skills       = models.JSONField(default=list, blank=True)
    languages         = models.JSONField(default=list, blank=True)
    awards            = models.JSONField(default=list, blank=True)
    created_at    = models.DateTimeField(auto_now_add=True)
    updated_at    = models.DateTimeField(auto_now=True)
    last_score    = models.FloatField(null=True, blank=True)
    class Meta:
        db_table = "resumes"
        ordering = ["-updated_at"]
    def __str__(self):
        return f"{self.user.email} - {self.name} v{self.version_number}"
    def to_text(self):
        parts = [self.full_name, self.email, self.summary]
        for e in self.experience:
            parts.append(f"{e.get('title','')} at {e.get('company','')} ({e.get('period','')})")
            for b in e.get("bullets", []):
                parts.append(f"- {b}")
        for e in self.education:
            parts.append(f"{e.get('degree','')} from {e.get('school','')} {e.get('year','')}")
        if self.technical_skills:
            parts.append(f"Technical: {', '.join(self.technical_skills)}")
        if self.soft_skills:
            parts.append(f"Soft: {', '.join(self.soft_skills)}")
        for p in self.projects:
            parts.append(f"Project: {p.get('name','')}: {p.get('desc','')}")
        if self.certifications:
            parts.append(f"Certs: {', '.join(str(c) for c in self.certifications)}")
        return "\n".join(filter(None, parts))

class ResumeFile(models.Model):
    resume         = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name="files")
    file           = models.FileField(upload_to="resume_files/")
    file_type      = models.CharField(max_length=10, default="pdf")
    extracted_text = models.TextField(blank=True)
    uploaded_at    = models.DateTimeField(auto_now_add=True)
    class Meta:
        db_table = "resume_files"