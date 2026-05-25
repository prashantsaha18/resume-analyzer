"""
ResumeAI Pro — ML Inference Engine
Loads trained models and serves predictions.
Falls back to Gemini if API key is set.
"""

import json
import re
import os
import logging
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional

import joblib
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"

# ─── Vocabulary (loaded from metadata) ────────────────────────────────────────
_meta = None

def _load_meta():
    global _meta
    if _meta is None:
        meta_path = MODELS_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                _meta = json.load(f)
        else:
            _meta = {
                "tech_skills": ["Python", "JavaScript", "TypeScript", "React", "Node.js", "Django",
                                 "FastAPI", "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes",
                                 "AWS", "GCP", "Azure", "Git", "CI/CD", "GraphQL", "REST API"],
                "strong_verbs": ["Led", "Built", "Engineered", "Architected", "Spearheaded", "Developed",
                                 "Designed", "Implemented", "Optimized", "Scaled", "Launched", "Delivered",
                                 "Transformed", "Automated", "Reduced", "Increased", "Improved", "Managed",
                                 "Mentored", "Collaborated", "Deployed", "Migrated", "Refactored", "Streamlined"],
                "weak_verbs": ["Responsible for", "Helped with", "Worked on", "Was involved in",
                               "Assisted with", "Participated in", "Did", "Made", "Tried to",
                               "Contributed to", "Part of the team that", "Helped", "Was part of"],
                "buzzwords": ["synergy", "leverage", "paradigm shift", "disruptive", "innovative",
                              "thought leader", "guru", "ninja", "rockstar", "wizard",
                              "holistic approach", "ecosystem", "bandwidth", "move the needle",
                              "circle back", "deep dive", "boil the ocean", "low-hanging fruit"],
                "passive_phrases": ["was responsible for managing", "was involved in the development of",
                                    "duties included the maintenance of", "helped in the creation of",
                                    "assisted in the implementation of", "tasks were completed related to"],
                "job_roles": ["Software Engineer", "Full Stack Developer", "Data Scientist"],
                "weakness_rules": ["no_quantification", "weak_action_verbs", "missing_summary"],
            }
        
        # Ensure top_companies and sections are present in _meta
        if "top_companies" not in _meta:
            _meta["top_companies"] = ["google", "meta", "amazon", "microsoft", "apple", "netflix", 
                                      "stripe", "airbnb", "uber", "linkedin", "twitter", "salesforce"]
        if "sections" not in _meta:
            _meta["sections"] = ['summary', 'experience', 'education', 'skills', 'projects', 
                                 'certifications', 'awards', 'publications', 'languages']
                                 
        # Cache pre-lowercased sets/lists for fast matching
        _meta["_tech_skills_lower"] = [s.lower() for s in _meta["tech_skills"]]
        _meta["_strong_verbs_lower"] = [v.lower() for v in _meta["strong_verbs"]]
        _meta["_weak_verbs_lower"] = [v.lower() for v in _meta["weak_verbs"]]
        _meta["_buzzwords_lower"] = [b.lower() for b in _meta["buzzwords"]]
        _meta["_passive_phrases_lower"] = [p.lower() for p in _meta["passive_phrases"]]
        _meta["_top_companies_lower"] = [c.lower() for c in _meta["top_companies"]]
        _meta["_sections_lower"] = [s.lower() for s in _meta["sections"]]
        
    return _meta


# ─── Model Loader ─────────────────────────────────────────────────────────────

class ModelStore:
    """Lazy-loads models on first use and caches them."""
    _loaded = {}

    @classmethod
    def get(cls, name):
        if name not in cls._loaded:
            path = MODELS_DIR / f"{name}.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"Model '{name}' not found. Run: python manage.py train_models"
                )
            cls._loaded[name] = joblib.load(path)
        return cls._loaded[name]

    @classmethod
    def available(cls) -> bool:
        return (MODELS_DIR / "tfidf_score.pkl").exists()


# ─── Feature Extractor ────────────────────────────────────────────────────────

def extract_features(text: str) -> np.ndarray:
    meta = _load_meta()
    
    # Use the pre-lowercased cached lists for O(1) lowercasing speed
    TECH_SKILLS = meta["_tech_skills_lower"]
    STRONG_VERBS = meta["_strong_verbs_lower"]
    WEAK_VERBS = meta["_weak_verbs_lower"]
    BUZZWORDS = meta["_buzzwords_lower"]
    PASSIVE_PHRASES = meta["_passive_phrases_lower"]
    SECTIONS = meta["_sections_lower"]
    TOP_COMPANIES = meta["_top_companies_lower"]

    text_lower = text.lower()
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!\n]', text) if len(s.strip()) > 5]

    features = []
    
    # 1-4. Length features
    features.append(len(words))
    features.append(len(text.split('\n')))
    features.append(len(sentences))
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    features.append(avg_sent_len)
    
    # 5-9. Formatting and digits
    features.append(text.count('•'))
    features.append(text.count('%'))
    features.append(sum(c.isdigit() for c in text)) # Optimized digit count
    features.append(len(re.findall(r'\$[\d,]+', text)))
    features.append(len(re.findall(r'\d+[KkMm]', text)))

    # 10-12. Action verbs (Optimized: direct boolean generator summing using cached lowercase lists)
    strong_count = sum(v in text_lower for v in STRONG_VERBS)
    weak_count = sum(v in text_lower for v in WEAK_VERBS)
    features.append(strong_count)
    features.append(weak_count)
    features.append(strong_count - weak_count)

    # 13-21. Section presence (binary)
    for sec in SECTIONS:
        features.append(int(sec in text_lower))

    # 22-25. Contact info (binary/regex)
    features.append(int('linkedin' in text_lower))
    features.append(int('github' in text_lower))
    features.append(int('@' in text))
    features.append(int(bool(re.search(r'\+?\d[\d\s\-()]{8,}', text))))

    # 26-27. Skill density
    skill_count = sum(s in text_lower for s in TECH_SKILLS)
    features.append(skill_count)
    features.append(skill_count / max(len(words), 1) * 100)

    # 28-31. Buzzwords, Passive voice, GPA
    buzzword_count = sum(b in text_lower for b in BUZZWORDS)
    passive_count = sum(p in text_lower for p in PASSIVE_PHRASES)
    features.append(buzzword_count)
    features.append(passive_count)
    features.append(int('gpa' in text_lower))
    features.append(int(bool(re.search(r'3\.[5-9]|4\.0', text))))

    # 32-33. Experience depth
    years_patterns = re.findall(r'20\d{2}', text)
    features.append(len(years_patterns))
    features.append(len(re.findall(r'present|current', text_lower)))

    # 34. Company prestige signal
    features.append(sum(c in text_lower for c in TOP_COMPANIES))

    return np.array(features, dtype=np.float32)


# ─── Core Inference Functions ──────────────────────────────────────────────────

def predict_scores(resume_text: str) -> dict:
    """Predict all 6 resume dimension scores using the ML model."""
    tfidf = ModelStore.get("tfidf_score")
    model = ModelStore.get("score_model")

    X_tfidf = tfidf.transform([resume_text]).toarray()
    hand_feat = extract_features(resume_text).reshape(1, -1)
    X = np.hstack([X_tfidf, hand_feat])

    preds = model.predict(X)[0]
    scores = {
        "ats_score":         int(np.clip(round(preds[0]), 10, 98)),
        "formatting_score":  int(np.clip(round(preds[1]), 10, 98)),
        "content_score":     int(np.clip(round(preds[2]), 10, 98)),
        "skills_score":      int(np.clip(round(preds[3]), 10, 98)),
        "keywords_score":    int(np.clip(round(preds[4]), 10, 98)),
        "readability_score": int(np.clip(round(preds[5]), 10, 98)),
    }
    return scores


def extract_keywords(resume_text: str) -> dict:
    """Extract found and missing keywords from resume."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]
    text_lower = resume_text.lower()

    found = [s for s in TECH_SKILLS if s.lower() in text_lower]

    # "Missing" = common skills not mentioned
    common_skills = ["Python", "JavaScript", "React", "Docker", "AWS", "Git",
                     "PostgreSQL", "REST API", "TypeScript", "CI/CD", "Linux",
                     "Kubernetes", "Redis", "GraphQL", "Node.js", "MongoDB"]
    missing = [s for s in common_skills if s.lower() not in text_lower][:8]

    return {"found_keywords": found, "missing_keywords": missing}


def detect_weaknesses(resume_text: str) -> dict:
    """Detect resume weaknesses using rule-based analysis."""
    meta = _load_meta()
    text_lower = resume_text.lower()
    words = resume_text.split()

    STRONG_VERBS = meta["strong_verbs"]
    WEAK_VERBS = meta["weak_verbs"]
    BUZZWORDS = meta["buzzwords"]
    PASSIVE_PHRASES = meta["passive_phrases"]
    TECH_SKILLS = meta["tech_skills"]

    issues = {}

    # Quantification
    digit_count = sum(1 for c in resume_text if c.isdigit())
    issues["no_quantification"] = digit_count < 5

    # Weak verbs
    weak_found = [v for v in WEAK_VERBS if v.lower() in text_lower]
    issues["weak_action_verbs"] = len(weak_found) > 0
    issues["weak_phrases"] = weak_found

    # Buzzwords
    buzz_found = [b for b in BUZZWORDS if b.lower() in text_lower]
    issues["buzzwords_overuse"] = len(buzz_found) > 2
    issues["buzzwords_found"] = buzz_found

    # Passive voice
    passive_found = [p for p in PASSIVE_PHRASES if p in text_lower]
    issues["passive_voice"] = len(passive_found) > 0
    issues["passive_instances"] = passive_found

    # Section checks
    issues["missing_summary"] = "summary" not in text_lower
    issues["missing_github"] = "github" not in text_lower
    issues["missing_linkedin"] = "linkedin" not in text_lower
    issues["missing_projects"] = "project" not in text_lower

    # Skills
    skill_count = sum(1 for s in TECH_SKILLS if s.lower() in text_lower)
    issues["few_skills"] = skill_count < 4

    # Length
    issues["short_resume"] = len(words) < 150
    issues["too_long"] = len(words) > 1200

    # Metrics
    has_metrics = any(c in resume_text for c in ['%', '$', 'K+', 'M+', 'million', 'thousand'])
    issues["no_metrics"] = not has_metrics

    # Strong verbs
    strong_count = sum(1 for v in STRONG_VERBS if v in resume_text)
    issues["low_action_verb_count"] = strong_count < 3

    # Generate human-readable weakness list
    weakness_list = []
    strength_list = []

    if issues["no_quantification"]:
        weakness_list.append("No quantified achievements — add numbers, percentages, and metrics")
    else:
        strength_list.append("Good use of quantified achievements with numbers and metrics")

    if issues["weak_action_verbs"]:
        weakness_list.append(f"Weak action verbs found: {', '.join(weak_found[:3])} — replace with power verbs")

    if issues["buzzwords_overuse"]:
        weakness_list.append(f"Overused buzzwords: {', '.join(buzz_found[:3])} — be more specific")

    if issues["passive_voice"]:
        weakness_list.append("Passive voice detected — use active voice for more impact")
    else:
        strength_list.append("Strong active voice throughout")

    if issues["missing_summary"]:
        weakness_list.append("Missing professional summary — add a 2-3 sentence intro")
    else:
        strength_list.append("Professional summary present")

    if issues["few_skills"]:
        weakness_list.append("Limited technical skills listed — add more relevant technologies")
    else:
        strength_list.append(f"Good range of technical skills ({skill_count} technologies mentioned)")

    if issues["no_metrics"]:
        weakness_list.append("No performance metrics — quantify your impact with numbers")

    if issues["short_resume"]:
        weakness_list.append("Resume is too brief — add more detail to experience section")

    if strong_count >= 5:
        strength_list.append(f"Excellent use of {strong_count} strong action verbs")

    if not issues["missing_github"]:
        strength_list.append("GitHub profile included — great for technical roles")

    return {
        "overall_weakness_score": len(weakness_list) * 10,
        "weaknesses": weakness_list,
        "strengths": strength_list,
        "issues_detail": issues,
        "stats": {
            "word_count": len(words),
            "skill_count": skill_count,
            "strong_verb_count": strong_count,
            "weak_verb_count": len(weak_found),
            "digit_count": digit_count,
        }
    }


def predict_job_roles(resume_text: str) -> dict:
    """Recommend job roles based on resume content."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]
    text_lower = resume_text.lower()

    role_skill_map = {
        "Software Engineer": ["Python", "JavaScript", "Java", "Git", "REST API", "SQL"],
        "Senior Software Engineer": ["Python", "System Design", "Microservices", "Docker", "Leadership"],
        "Full Stack Developer": ["React", "Node.js", "PostgreSQL", "REST API", "TypeScript"],
        "Frontend Engineer": ["React", "Vue.js", "TypeScript", "CSS", "Next.js", "Angular"],
        "Backend Engineer": ["Python", "Java", "Go", "PostgreSQL", "Redis", "Kafka"],
        "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Terraform", "AWS", "Linux"],
        "Data Scientist": ["Python", "Machine Learning", "Pandas", "NumPy", "TensorFlow", "SQL"],
        "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Docker"],
        "Cloud Architect": ["AWS", "GCP", "Azure", "Terraform", "Kubernetes", "Docker"],
        "Engineering Manager": ["Leadership", "Agile", "Scrum", "Mentoring", "Communication"],
        "SRE Engineer": ["Linux", "Docker", "Kubernetes", "Python", "Monitoring", "CI/CD"],
        "Mobile Developer": ["React", "Swift", "Kotlin", "JavaScript", "API"],
    }

    salary_map = {
        "Software Engineer": "$80K – $130K",
        "Senior Software Engineer": "$130K – $200K",
        "Full Stack Developer": "$90K – $150K",
        "Frontend Engineer": "$85K – $145K",
        "Backend Engineer": "$90K – $160K",
        "DevOps Engineer": "$100K – $170K",
        "Data Scientist": "$95K – $165K",
        "ML Engineer": "$120K – $200K",
        "Cloud Architect": "$130K – $210K",
        "Engineering Manager": "$150K – $230K",
        "SRE Engineer": "$110K – $180K",
        "Mobile Developer": "$85K – $150K",
    }

    recommendations = []
    for role, skills in role_skill_map.items():
        matched = [s for s in skills if s.lower() in text_lower]
        match_pct = int((len(matched) / len(skills)) * 100)
        if match_pct > 20:
            missing = [s for s in skills if s.lower() not in text_lower][:3]
            recommendations.append({
                "title": role,
                "match_percentage": match_pct,
                "why_good_fit": f"Your resume matches {len(matched)}/{len(skills)} key requirements for this role.",
                "salary_range": salary_map.get(role, "$80K – $150K"),
                "required_upskilling": missing,
                "matched_skills": matched,
            })

    recommendations.sort(key=lambda x: x["match_percentage"], reverse=True)

    return {
        "recommendations": recommendations[:8],
        "career_trajectory": {
            "current_level": _estimate_level(resume_text),
            "next_step": recommendations[1]["title"] if len(recommendations) > 1 else "Senior Engineer",
            "long_term_path": "Tech Lead → Engineering Manager → VP Engineering",
        },
        "strongest_industries": _detect_industries(text_lower),
    }


def _estimate_level(text: str) -> str:
    text_lower = text.lower()
    years = re.findall(r'(\d+)\+?\s+years?', text_lower)
    max_years = max([int(y) for y in years], default=0) if years else 0
    if max_years >= 8 or "senior" in text_lower or "lead" in text_lower or "manager" in text_lower:
        return "Senior / Lead"
    elif max_years >= 3 or "engineer" in text_lower:
        return "Mid-Level"
    else:
        return "Junior / Entry-Level"


def _detect_industries(text_lower: str) -> list:
    industries = []
    if any(w in text_lower for w in ['fintech', 'payment', 'banking', 'finance', 'stripe', 'plaid']):
        industries.append("FinTech")
    if any(w in text_lower for w in ['health', 'medical', 'clinical', 'healthcare', 'ehr']):
        industries.append("HealthTech")
    if any(w in text_lower for w in ['ml', 'machine learning', 'ai', 'data science', 'deep learning']):
        industries.append("AI / ML")
    if any(w in text_lower for w in ['cloud', 'aws', 'azure', 'gcp', 'infrastructure']):
        industries.append("Cloud / Infrastructure")
    if any(w in text_lower for w in ['react', 'frontend', 'ui', 'ux', 'design']):
        industries.append("Product / Frontend")
    if not industries:
        industries = ["Software Engineering", "Technology"]
    return industries[:3]


def compute_ats_match(resume_text: str, job_description: str) -> dict:
    """Compute ATS keyword match between resume and JD."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]

    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # Extract skills mentioned in JD
    jd_skills = [s for s in TECH_SKILLS if s.lower() in jd_lower]
    resume_skills = [s for s in TECH_SKILLS if s.lower() in resume_lower]

    if not jd_skills:
        # Fall back to word overlap
        jd_words = set(re.findall(r'\b\w{4,}\b', jd_lower))
        resume_words = set(re.findall(r'\b\w{4,}\b', resume_lower))
        overlap = jd_words & resume_words
        match_pct = min(95, int(len(overlap) / max(len(jd_words), 1) * 150))
        return {
            "match_score": match_pct,
            "keyword_match_percentage": match_pct,
            "matched_keywords": list(overlap)[:10],
            "missing_critical_keywords": list(jd_words - resume_words)[:8],
            "missing_nice_to_have": [],
            "recommendation": "Apply" if match_pct >= 60 else "Apply with modifications",
            "reasoning": f"Your resume overlaps {match_pct}% with the job description vocabulary.",
            "optimization_tips": ["Add more keywords from the job description to your skills section"],
        }

    matched = [s for s in jd_skills if s.lower() in resume_lower]
    missing = [s for s in jd_skills if s.lower() not in resume_lower]
    match_pct = int((len(matched) / max(len(jd_skills), 1)) * 100)

    if match_pct >= 75:
        recommendation = "Strong Apply"
        reasoning = f"Excellent match! You meet {len(matched)}/{len(jd_skills)} key skill requirements."
    elif match_pct >= 50:
        recommendation = "Apply"
        reasoning = f"Good match. You meet {len(matched)}/{len(jd_skills)} requirements. Address the gaps if possible."
    elif match_pct >= 30:
        recommendation = "Apply with Modifications"
        reasoning = f"Partial match ({match_pct}%). Add missing skills to your profile before applying."
    else:
        recommendation = "Skill Gap Too Large"
        reasoning = f"Only {match_pct}% match. Consider upskilling before applying."

    tips = []
    if missing:
        tips.append(f"Add these skills to your resume: {', '.join(missing[:3])}")
    tips.append("Tailor your summary to mirror the job description language")
    tips.append("Add a 'Key Skills' section matching the JD requirements exactly")

    return {
        "match_score": match_pct,
        "keyword_match_percentage": match_pct,
        "matched_keywords": matched,
        "missing_critical_keywords": missing[:6],
        "missing_nice_to_have": [s for s in resume_skills if s not in jd_skills][:4],
        "recommendation": recommendation,
        "reasoning": reasoning,
        "optimization_tips": tips,
    }


def analyze_skill_gap(resume_text: str, job_description: str) -> dict:
    """Detailed skill gap analysis between resume and JD."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    jd_skills = [s for s in TECH_SKILLS if s.lower() in jd_lower]
    resume_skills = [s for s in TECH_SKILLS if s.lower() in resume_lower]

    matching = [s for s in jd_skills if s.lower() in resume_lower]
    critical_missing = [s for s in jd_skills if s.lower() not in resume_lower]
    nice_to_have = [s for s in TECH_SKILLS if s.lower() in jd_lower and s not in jd_skills]

    readiness = int((len(matching) / max(len(jd_skills), 1)) * 100)

    learning_path = []
    for skill in critical_missing[:4]:
        resources = {
            "Python": ["Official Docs", "Automate the Boring Stuff (free)", "Real Python"],
            "React": ["React Docs", "Scrimba React Course", "Full Stack Open"],
            "Docker": ["Docker Getting Started", "Play with Docker", "Docker Mastery on Udemy"],
            "AWS": ["AWS Free Tier", "A Cloud Guru", "AWS Certified Developer Course"],
            "Kubernetes": ["Kubernetes.io Docs", "KodeKloud", "CKA Course"],
            "TypeScript": ["TypeScript Handbook", "Execute Program", "Matt Pocock's TS tutorials"],
        }.get(skill, ["Official Documentation", "YouTube Tutorials", "Udemy Beginner Course"])

        learning_path.append({
            "skill": skill,
            "resources": resources[:2],
            "estimated_time": "2-4 weeks" if skill in ["Python", "React", "TypeScript"] else "4-8 weeks",
        })

    return {
        "matching_skills": matching,
        "critical_missing_skills": critical_missing,
        "nice_to_have_missing": nice_to_have[:4],
        "transferable_skills": [s for s in resume_skills if s not in jd_skills][:5],
        "learning_path": learning_path,
        "overall_readiness": readiness,
        "summary": f"You match {len(matching)}/{len(jd_skills)} required skills ({readiness}% readiness). "
                   f"Focus on: {', '.join(critical_missing[:3]) if critical_missing else 'You have all key skills!'}",
    }


def score_bullet(bullet: str) -> dict:
    """Score and improve a single resume bullet point."""
    meta = _load_meta()
    STRONG_VERBS = meta["strong_verbs"]
    WEAK_VERBS = meta["weak_verbs"]
    bullet_lower = bullet.lower()

    # Analyze issues
    has_number = bool(re.search(r'\d', bullet))
    has_strong_verb = any(v.lower() in bullet_lower for v in STRONG_VERBS)
    has_weak_verb = any(v.lower() in bullet_lower for v in WEAK_VERBS)
    word_count = len(bullet.split())

    issues = []
    if not has_number:
        issues.append("No quantification — add a metric or number")
    if has_weak_verb:
        issues.append("Weak action verb — replace with a power verb")
    if not has_strong_verb and not has_weak_verb:
        issues.append("No clear action verb at the start")
    if word_count < 8:
        issues.append("Too brief — add more context and impact")
    if word_count > 30:
        issues.append("Too long — keep bullets concise (10-20 words ideal)")

    # Generate alternatives
    strong_verb = STRONG_VERBS[hash(bullet) % len(STRONG_VERBS)]
    strong_verb2 = STRONG_VERBS[(hash(bullet) + 3) % len(STRONG_VERBS)]
    strong_verb3 = STRONG_VERBS[(hash(bullet) + 7) % len(STRONG_VERBS)]

    core = re.sub(r'^(responsible for|helped|worked on|was involved in|assisted with|participated in)\s*', '', bullet_lower).strip()
    core = core[0].upper() + core[1:] if core else bullet

    alternatives = [
        {
            "text": f"{strong_verb} {core}, resulting in measurable performance improvement",
            "reason": "Added strong action verb + result framing"
        },
        {
            "text": f"{strong_verb2} {core} reducing time/cost by X% through targeted optimization",
            "reason": "Added quantification placeholder and optimization context"
        },
        {
            "text": f"{strong_verb3} {core} that improved team productivity and delivery speed",
            "reason": "Added impact statement with team context"
        },
    ]

    base_score = 40
    if has_number: base_score += 25
    if has_strong_verb: base_score += 20
    if 10 <= word_count <= 25: base_score += 10
    if not has_weak_verb: base_score += 5

    return {
        "original_score": min(95, base_score),
        "issues_with_original": issues,
        "alternatives": alternatives,
        "techniques_used": ["Strong action verbs", "CAR format (Challenge-Action-Result)", "Quantification"],
    }


def check_grammar(resume_text: str) -> dict:
    """Analyze grammar and readability of resume."""
    meta = _load_meta()
    PASSIVE_PHRASES = meta["passive_phrases"]
    WEAK_VERBS = meta["weak_verbs"]

    lines = [l.strip() for l in resume_text.split('\n') if len(l.strip()) > 10]
    words = resume_text.split()
    sentences = [s.strip() for s in re.split(r'[.!\n]', resume_text) if len(s.strip()) > 10]

    # Passive voice
    passive_instances = []
    for line in lines:
        if any(p in line.lower() for p in PASSIVE_PHRASES):
            active = re.sub(r'(was|were|is|are)\s+(responsible for|involved in)', 'managed', line, flags=re.I)
            passive_instances.append({"original": line[:80], "active_version": active[:80]})

    # Long sentences
    long_sentences = [s for s in sentences if len(s.split()) > 25]

    # Tense consistency check (bullets should be past tense except current role)
    consistency_issues = []
    present_verbs = ['manage', 'lead', 'develop', 'build', 'create', 'maintain', 'work']
    past_verbs = ['managed', 'led', 'developed', 'built', 'created', 'maintained', 'worked']
    bullet_lines = [l for l in lines if l.startswith('•')]
    if bullet_lines:
        has_present = any(any(v in l.lower() for v in present_verbs) for l in bullet_lines)
        has_past = any(any(v in l.lower() for v in past_verbs) for l in bullet_lines)
        if has_present and has_past:
            consistency_issues.append("Mixed tenses in bullet points — use past tense for previous roles")

    # Readability score
    avg_words = np.mean([len(s.split()) for s in sentences]) if sentences else 15
    readability = max(30, min(95, 95 - max(0, avg_words - 15) * 2 - len(passive_instances) * 5))

    # Grade
    grade = "A" if readability >= 85 else "B" if readability >= 70 else "C" if readability >= 55 else "D"

    return {
        "grammar_errors": [],  # rule-based, no errors found by default
        "passive_voice_instances": passive_instances[:4],
        "long_sentences": [s[:100] for s in long_sentences[:3]],
        "readability_score": int(readability),
        "reading_level": "Professional" if readability >= 75 else "General",
        "consistency_issues": consistency_issues,
        "formatting_issues": _check_formatting(resume_text),
        "overall_grade": grade,
        "summary": f"Overall readability grade {grade} ({int(readability)}/100). "
                   f"Found {len(passive_instances)} passive voice instances and {len(long_sentences)} long sentences.",
    }


def _check_formatting(text: str) -> list:
    issues = []
    lines = text.split('\n')
    dates = re.findall(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February)\s*\d{4}|\d{4}\s*[-–]\s*(\d{4}|Present)', text)
    if not dates and re.findall(r'\d{4}', text):
        issues.append("Consider standardizing date formats (e.g., Jan 2020 – Present)")
    return issues


def generate_cover_letter_ml(resume_text: str, company: str, role: str, tone: str, job_description: str = "") -> str:
    """Generate a structured cover letter using templates + ML extraction."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]
    text_lower = resume_text.lower()

    # Extract candidate name (first line usually)
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
    name = lines[0] if lines else "I"

    # Extract top skills
    skills = [s for s in TECH_SKILLS if s.lower() in text_lower][:4]
    skills_str = ", ".join(skills) if skills else "software development"

    # Extract years of experience
    years_matches = re.findall(r'(\d+)\+?\s+years?', text_lower)
    years = max([int(y) for y in years_matches], default=0) if years_matches else 0
    years_str = f"{years}+ years of" if years > 0 else "extensive"

    # Extract most recent role
    exp_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\|\s*([A-Z][A-Za-z\s&]+)\s*\|', resume_text)
    recent_role = exp_match.group(1) if exp_match else "software engineer"
    recent_company = exp_match.group(2).strip() if exp_match else "my previous employer"

    # JD keyword extraction
    jd_keywords = []
    if job_description:
        jd_keywords = [s for s in TECH_SKILLS if s.lower() in job_description.lower()][:3]

    tone_openers = {
        "professional": f"I am writing to express my strong interest in the {role} position at {company}.",
        "enthusiastic": f"The {role} opportunity at {company} immediately caught my attention — it aligns perfectly with both my technical expertise and career ambitions.",
        "formal": f"I respectfully submit my application for the position of {role} at {company}, as advertised.",
        "conversational": f"When I came across the {role} role at {company}, I knew I had to apply.",
        "concise": f"I am an experienced {recent_role} applying for the {role} position at {company}.",
    }

    opener = tone_openers.get(tone, tone_openers["professional"])
    skills_mention = f"React, Node.js, and PostgreSQL" if not jd_keywords else ", ".join(jd_keywords)

    cover_letter = f"""{name}
{company} Hiring Team

Dear Hiring Manager,

{opener}

With {years_str} experience as a {recent_role}, I have developed deep expertise in {skills_str}. At {recent_company}, I delivered scalable solutions that drove measurable business impact — from optimizing system performance to leading cross-functional engineering initiatives. I am particularly skilled in {skills_mention}, which I understand are central to this role at {company}.

What draws me to {company} specifically is your reputation for engineering excellence and innovation. I am confident that my background in building high-performance, production-grade systems would allow me to contribute meaningfully to your team from day one. I thrive in collaborative environments and have a track record of mentoring peers, improving engineering processes, and delivering projects on time and under budget.

I would welcome the opportunity to discuss how my experience aligns with your needs. Thank you for considering my application — I look forward to the possibility of contributing to {company}'s continued success.

Best regards,
{name}"""

    return cover_letter


def generate_portfolio_structure(resume_text: str) -> dict:
    """Extract structured data from resume for portfolio generation."""
    meta = _load_meta()
    TECH_SKILLS = meta["tech_skills"]
    text_lower = resume_text.lower()
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]

    name = lines[0] if lines else "Developer"
    skills = [s for s in TECH_SKILLS if s.lower() in text_lower][:10]
    exp_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*\|\s*([A-Z][A-Za-z\s&]+)', resume_text)
    role = exp_match.group(1) if exp_match else "Software Engineer"

    return {
        "name": name,
        "role": role,
        "skills": skills,
        "summary": _extract_section(resume_text, "summary"),
        "experience": _extract_section(resume_text, "experience"),
    }


def _extract_section(text: str, section: str) -> str:
    """Extract a section from resume text."""
    lines = text.split('\n')
    in_section = False
    result = []
    for line in lines:
        if section.upper() in line.upper() and len(line.strip()) < 30:
            in_section = True
            continue
        if in_section:
            if line.isupper() and len(line.strip()) > 2 and section.upper() not in line.upper():
                break
            result.append(line)
    return '\n'.join(result[:8]).strip()


# ─── Main Analysis Orchestrator ────────────────────────────────────────────────

def full_analysis(resume_text: str, job_description: str = "") -> dict:
    """Run complete resume analysis using ML models."""
    if not ModelStore.available():
        raise RuntimeError("ML models not trained. Run: python manage.py train_models")

    scores = predict_scores(resume_text)
    keywords = extract_keywords(resume_text)
    weakness_data = detect_weaknesses(resume_text)

    overall = (
        scores['ats_score'] * 0.30 +
        scores['formatting_score'] * 0.15 +
        scores['content_score'] * 0.25 +
        scores['skills_score'] * 0.15 +
        scores['keywords_score'] * 0.10 +
        scores['readability_score'] * 0.05
    )

    return {
        **scores,
        "found_keywords": keywords["found_keywords"],
        "missing_keywords": keywords["missing_keywords"],
        "strengths": weakness_data["strengths"][:4],
        "weaknesses": weakness_data["weaknesses"][:4],
        "suggestions": _generate_suggestions(scores, weakness_data),
        "grammar_issues": [],
        "action_verbs": {
            "strong": [v for v in meta_verbs(True) if v.lower() in resume_text.lower()][:5],
            "weak": [v for v in meta_verbs(False) if v.lower() in resume_text.lower()][:3],
        },
        "overall_feedback": (
            f"Your resume scores {int(overall)}/100 overall. "
            f"{'Strong ATS compatibility.' if scores['ats_score'] >= 70 else 'Needs ATS optimization.'} "
            f"{'Good content quality.' if scores['content_score'] >= 65 else 'Improve content with quantified achievements.'} "
            f"Focus on: {', '.join(keywords['missing_keywords'][:3]) if keywords['missing_keywords'] else 'maintaining current quality'}."
        ),
    }


def meta_verbs(strong: bool) -> list:
    meta = _load_meta()
    return meta["strong_verbs"] if strong else meta["weak_verbs"]


def _generate_suggestions(scores: dict, weakness_data: dict) -> list:
    suggestions = []
    if scores['ats_score'] < 70:
        suggestions.append("Add industry-standard keywords from job descriptions to improve ATS score")
    if scores['content_score'] < 65:
        suggestions.append("Quantify achievements with numbers, percentages, and impact metrics")
    if scores['formatting_score'] < 70:
        suggestions.append("Add missing sections: summary, projects, or certifications")
    if weakness_data['issues_detail'].get('weak_action_verbs'):
        suggestions.append("Replace weak phrases like 'responsible for' with power verbs: Led, Built, Engineered")
    if weakness_data['issues_detail'].get('no_quantification'):
        suggestions.append("Include specific metrics: team size, performance gains, user counts, cost savings")
    if weakness_data['issues_detail'].get('missing_github'):
        suggestions.append("Add your GitHub profile URL to showcase your code and projects")
    if len(suggestions) < 4:
        suggestions.append("Tailor your resume summary to match the target job description")
    return suggestions[:5]
