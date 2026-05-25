"""
ResumeAI Pro — ML Model Trainer
Trains 8 specialized models for resume analysis.
Run via: python manage.py train_models
"""

import os
import json
import pickle
import numpy as np
import re
from pathlib import Path

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from .data_generator import (
    generate_dataset, generate_jd_resume_pairs,
    TECH_SKILLS, STRONG_VERBS, WEAK_VERBS, BUZZWORDS, PASSIVE_PHRASES,
    JOB_ROLES
)

MODELS_DIR = Path(__file__).parent / "models"


# ─── Feature Engineering ───────────────────────────────────────────────────────

from ml_engine.inference import extract_features


def prepare_training_data(dataset):
    """Convert dataset to X (features) and y (scores) matrices."""
    texts = [d['text'] for d in dataset]
    hand_features = np.array([extract_features(d['text']) for d in dataset])

    y_ats      = np.array([d['scores']['ats_score'] for d in dataset])
    y_fmt      = np.array([d['scores']['formatting_score'] for d in dataset])
    y_content  = np.array([d['scores']['content_score'] for d in dataset])
    y_skills   = np.array([d['scores']['skills_score'] for d in dataset])
    y_keywords = np.array([d['scores']['keywords_score'] for d in dataset])
    y_read     = np.array([d['scores']['readability_score'] for d in dataset])

    # Multi-output target
    y_all = np.column_stack([y_ats, y_fmt, y_content, y_skills, y_keywords, y_read])

    return texts, hand_features, y_all, y_ats


# ─── Model 1: Multi-Score Regressor ───────────────────────────────────────────

def train_score_model(texts, hand_features, y_all):
    """
    Train the main scoring model.
    Uses TF-IDF + hand features + GradientBoosting for each output.
    """
    print("\n[1/8] Training Multi-Score Regressor...")

    is_quick = len(texts) <= 1000
    max_feats = 300 if is_quick else 3000
    n_est = 30 if is_quick else 200
    max_d = 3 if is_quick else 5

    # TF-IDF on text
    tfidf = TfidfVectorizer(
        max_features=max_feats,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2 if not is_quick else 1,
        strip_accents='unicode',
    )
    X_tfidf = tfidf.fit_transform(texts).toarray()

    # Combine TF-IDF + hand features
    X = np.hstack([X_tfidf, hand_features])

    X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=0.15, random_state=42)

    model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=4 if not is_quick else 2,
            random_state=42,
        ),
        n_jobs=None
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"   MAE: {mae:.2f} points | Targets: [ATS, Fmt, Content, Skills, Keywords, Readability]")

    for i, name in enumerate(['ATS', 'Formatting', 'Content', 'Skills', 'Keywords', 'Readability']):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"   {name}: R²={r2:.3f}, MAE={mean_absolute_error(y_test[:,i], y_pred[:,i]):.1f}")

    return tfidf, model


# ─── Model 2: Keyword Extractor ───────────────────────────────────────────────

def train_keyword_model(texts):
    """
    Train keyword extraction model using TF-IDF with tech skill vocabulary.
    """
    print("\n[2/8] Training Keyword Extractor...")

    # Build a vocabulary focused on tech skills + career terms
    vocab_terms = [s.lower() for s in TECH_SKILLS]
    vocab_terms += ['leadership', 'management', 'agile', 'scrum', 'devops',
                    'machine learning', 'deep learning', 'neural network',
                    'cloud', 'microservices', 'api', 'database', 'testing',
                    'deployment', 'architecture', 'performance', 'security']

    keyword_vectorizer = TfidfVectorizer(
        vocabulary={term: i for i, term in enumerate(set(vocab_terms))},
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    keyword_vectorizer.fit(texts)
    print(f"   Vocabulary size: {len(keyword_vectorizer.vocabulary_)} terms")
    return keyword_vectorizer


# ─── Model 3: Job Role Classifier ─────────────────────────────────────────────

def train_job_role_model(dataset):
    """
    Classify resume into recommended job roles using multi-label classification.
    """
    print("\n[3/8] Training Job Role Classifier...")

    role_skill_map = {
        "Software Engineer": ["Python", "JavaScript", "Java", "Git", "REST API", "SQL"],
        "Senior Software Engineer": ["Python", "JavaScript", "System Design", "Microservices", "Docker"],
        "Full Stack Developer": ["React", "Node.js", "PostgreSQL", "REST API", "TypeScript"],
        "Frontend Engineer": ["React", "Vue.js", "TypeScript", "CSS", "Next.js", "Angular"],
        "Backend Engineer": ["Python", "Java", "Go", "PostgreSQL", "Redis", "Kafka"],
        "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Terraform", "AWS", "Linux"],
        "Data Scientist": ["Python", "Machine Learning", "Pandas", "NumPy", "TensorFlow", "SQL"],
        "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Docker", "MLOps"],
        "Cloud Architect": ["AWS", "GCP", "Azure", "Terraform", "Kubernetes", "Docker"],
        "Engineering Manager": ["Leadership", "Agile", "Scrum", "System Design", "Mentoring"],
    }

    X_texts = []
    y_roles = []

    for sample in dataset:
        text_lower = sample['text'].lower()
        applicable_roles = []
        for role, req_skills in role_skill_map.items():
            match_count = sum(1 for s in req_skills if s.lower() in text_lower)
            if match_count >= 3:
                applicable_roles.append(role)
        if not applicable_roles:
            applicable_roles = ["Software Engineer"]
        X_texts.append(sample['text'])
        y_roles.append(applicable_roles)

    is_quick = len(dataset) <= 1000
    max_feats = 200 if is_quick else 2000
    n_est = 20 if is_quick else 100
    max_d = 4 if is_quick else 8

    mlb = MultiLabelBinarizer(classes=list(role_skill_map.keys()))
    y_encoded = mlb.fit_transform(y_roles)

    tfidf = TfidfVectorizer(max_features=max_feats, ngram_range=(1, 2), sublinear_tf=True, min_df=2 if not is_quick else 1)
    X = tfidf.fit_transform(X_texts)

    clf = MultiOutputRegressor(
        RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=1),
        n_jobs=None
    )
    clf.fit(X, y_encoded)
    print(f"   Trained on {len(X_texts)} samples, {len(mlb.classes_)} roles")
    return tfidf, clf, mlb


# ─── Model 4: Weakness Detector ───────────────────────────────────────────────

def train_weakness_model(texts):
    """
    Rule-based + ML hybrid weakness detection.
    Returns weakness categories for a resume.
    """
    print("\n[4/8] Training Weakness Detector...")

    weakness_rules = {
        "no_quantification": lambda t: sum(1 for c in t if c.isdigit()) < 5,
        "weak_action_verbs": lambda t: sum(1 for v in WEAK_VERBS if v.lower() in t.lower()) > 0,
        "buzzwords_overuse": lambda t: sum(1 for b in BUZZWORDS if b.lower() in t.lower()) > 2,
        "passive_voice": lambda t: sum(1 for p in PASSIVE_PHRASES if p in t.lower()) > 0,
        "missing_summary": lambda t: 'summary' not in t.lower(),
        "missing_github": lambda t: 'github' not in t.lower(),
        "few_skills": lambda t: sum(1 for s in TECH_SKILLS if s.lower() in t.lower()) < 4,
        "short_resume": lambda t: len(t.split()) < 150,
        "no_metrics": lambda t: not any(c in t for c in ['%', '$', 'K+', 'M+', 'million', 'thousand']),
        "vague_language": lambda t: any(phrase in t.lower() for phrase in ['various', 'etc', 'and so on', 'multiple tasks']),
    }

    # Fit a simple vectorizer for text patterns
    vect = TfidfVectorizer(max_features=1000, ngram_range=(1, 3), sublinear_tf=True)
    vect.fit(texts)

    print(f"   {len(weakness_rules)} weakness categories configured")
    return vect, weakness_rules


# ─── Model 5: ATS Match Scorer ────────────────────────────────────────────────

def train_ats_match_model(pairs):
    """
    Train ATS keyword match model on resume+JD pairs.
    """
    print("\n[5/8] Training ATS Match Model...")

    X_texts = [p['resume_text'] + " [SEP] " + p['job_description'] for p in pairs]
    y_match = np.array([p['match_score'] for p in pairs])

    is_quick = len(pairs) <= 1000
    max_feats = 400 if is_quick else 4000
    n_est = 30 if is_quick else 150
    max_d = 3 if is_quick else 4

    tfidf = TfidfVectorizer(
        max_features=max_feats,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X = tfidf.fit_transform(X_texts)

    model = GradientBoostingRegressor(
        n_estimators=n_est,
        max_depth=max_d,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y_match)

    # Quick eval
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_match, test_size=0.1, random_state=42)
    model2 = GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)
    model2.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, model2.predict(X_te))
    print(f"   MAE: {mae:.1f}% on match score")

    return tfidf, model


# ─── Model 6: Grammar / Readability Scorer ────────────────────────────────────

def train_readability_model(texts, y_readability):
    """
    Train readability scorer using linguistic features.
    """
    print("\n[6/8] Training Readability Model...")

    hand_features = np.array([extract_features(t) for t in texts])

    X_train, X_test, y_train, y_test = train_test_split(
        hand_features, y_readability, test_size=0.15, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=0.5))
    ])
    pipe.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, pipe.predict(X_test))
    print(f"   MAE: {mae:.1f} readability points")
    return pipe


# ─── Model 7: Bullet Quality Scorer ───────────────────────────────────────────

def train_bullet_scorer():
    """
    Rule-based + learned bullet quality scorer.
    Scores individual bullet points 0-100.
    """
    print("\n[7/8] Training Bullet Quality Scorer...")

    bullet_samples = []
    for _ in range(2000):
        quality = np.random.choice(['high', 'medium', 'low', 'weak'], p=[0.3, 0.3, 0.25, 0.15])
        from .data_generator import generate_quantified_bullet, STRONG_VERBS
        verb = np.random.choice(STRONG_VERBS)
        bullet = generate_quantified_bullet(verb, quality)
        score = {'high': np.random.randint(78, 98),
                 'medium': np.random.randint(48, 72),
                 'low': np.random.randint(25, 50),
                 'weak': np.random.randint(10, 30)}[quality]
        bullet_samples.append((bullet, score))

    X_texts = [b[0] for b in bullet_samples]
    y_scores = np.array([b[1] for b in bullet_samples])

    vect = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vect.fit_transform(X_texts)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y_scores)
    print(f"   Trained on 2000 bullet samples")
    return vect, model


# ─── Model 8: Skill Gap Classifier ────────────────────────────────────────────

def build_skill_gap_analyzer():
    """
    Skill gap analyzer using TF-IDF cosine similarity.
    No training needed — pure similarity computation.
    """
    print("\n[8/8] Building Skill Gap Analyzer...")
    vect = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    print("   Skill gap analyzer configured (cosine similarity)")
    return vect


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train_all_models(n_samples=5000):
    """
    Main entry point: train all 8 models and save to disk.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("  ResumeAI Pro — ML Training Pipeline")
    print(f"  Training on {n_samples} synthetic resume samples")
    print("=" * 55)

    # Generate data
    print("\n📊 Generating training data...")
    dataset = generate_dataset(n_samples)
    print(f"✓ {len(dataset)} resume samples ready")

    print("\n📊 Generating JD-resume pairs for ATS matching...")
    jd_pairs = generate_jd_resume_pairs(1000)
    print(f"✓ {len(jd_pairs)} JD-resume pairs ready")

    texts, hand_features, y_all, y_ats = prepare_training_data(dataset)
    y_readability = y_all[:, 5]

    # Train all models
    tfidf_score, score_model = train_score_model(texts, hand_features, y_all)
    keyword_vect = train_keyword_model(texts)
    role_tfidf, role_model, mlb = train_job_role_model(dataset)
    weakness_vect, weakness_rules = train_weakness_model(texts)
    ats_tfidf, ats_model = train_ats_match_model(jd_pairs)
    readability_model = train_readability_model(texts, y_readability)
    bullet_vect, bullet_model = train_bullet_scorer()
    skill_gap_vect = build_skill_gap_analyzer()
    skill_gap_vect.fit(texts)

    # Save all models
    print("\n💾 Saving models to disk...")
    artifacts = {
        "tfidf_score":       tfidf_score,
        "score_model":       score_model,
        "keyword_vect":      keyword_vect,
        "role_tfidf":        role_tfidf,
        "role_model":        role_model,
        "role_mlb":          mlb,
        "weakness_vect":     weakness_vect,
        "weakness_rules":    None,          # rules are code-based, not saved
        "ats_tfidf":         ats_tfidf,
        "ats_model":         ats_model,
        "readability_model": readability_model,
        "bullet_vect":       bullet_vect,
        "bullet_model":      bullet_model,
        "skill_gap_vect":    skill_gap_vect,
    }

    for name, obj in artifacts.items():
        if obj is not None:
            path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(obj, path)
            size_kb = path.stat().st_size // 1024
            print(f"   ✓ {name}.pkl ({size_kb} KB)")

    # Save weakness rules as metadata
    rules_meta = {
        "weakness_rules": list(weakness_rules.keys()),
        "tech_skills": TECH_SKILLS,
        "strong_verbs": STRONG_VERBS,
        "weak_verbs": WEAK_VERBS,
        "buzzwords": BUZZWORDS,
        "passive_phrases": list(PASSIVE_PHRASES),
        "job_roles": JOB_ROLES,
        "top_companies": ["google", "meta", "amazon", "microsoft", "apple", "netflix", 
                          "stripe", "airbnb", "uber", "linkedin", "twitter", "salesforce"],
        "sections": ["summary", "experience", "education", "skills", "projects", 
                     "certifications", "awards", "publications", "languages"],
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(rules_meta, f, indent=2)
    print("   ✓ metadata.json")

    print("\n" + "=" * 55)
    print("  ✅ All 8 models trained and saved successfully!")
    print(f"  📁 Models directory: {MODELS_DIR}")
    print("=" * 55)

    return artifacts
