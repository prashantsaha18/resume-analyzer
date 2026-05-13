"""
Lightweight inference — works on Vercel without sklearn.
Called automatically when .pkl model files are not present.
"""
import re
import numpy as np
from ml_engine.inference import (
    _load_meta, extract_keywords, detect_weaknesses,
    compute_ats_match, analyze_skill_gap, score_bullet,
    check_grammar, predict_job_roles, generate_cover_letter_ml,
    generate_portfolio_structure, _extract_section
)


def predict_scores_lite(resume_text: str) -> dict:
    """
    Rule-based scoring when ML models are not available.
    Produces realistic scores using hand-crafted heuristics.
    """
    meta   = _load_meta()
    SKILLS = meta['tech_skills']
    STRONG = meta['strong_verbs']
    WEAK   = meta['weak_verbs']
    BUZZ   = meta['buzzwords']

    text_lower = resume_text.lower()
    words      = resume_text.split()

    # Section presence
    has_summary  = 'summary'        in text_lower
    has_exp      = 'experience'     in text_lower
    has_edu      = 'education'      in text_lower
    has_skills   = 'skill'          in text_lower
    has_projects = 'project'        in text_lower
    has_certs    = 'certif'         in text_lower
    has_linkedin = 'linkedin'       in text_lower
    has_github   = 'github'         in text_lower

    # Counts
    bullet_count  = resume_text.count('•')
    digit_count   = sum(1 for c in resume_text if c.isdigit())
    skill_count   = sum(1 for s in SKILLS if s.lower() in text_lower)
    strong_count  = sum(1 for v in STRONG if v.lower() in text_lower)
    weak_count    = sum(1 for v in WEAK   if v.lower() in text_lower)
    buzz_count    = sum(1 for b in BUZZ   if b.lower() in text_lower)
    has_metrics   = any(c in resume_text for c in ['%', '$', 'K+', 'M+'])

    # ATS Score (30-95 range)
    ats = 30
    ats += min(20, skill_count * 3)     # skills presence
    ats += 10 if has_summary else 0
    ats += 8  if has_linkedin else 0
    ats += 8  if has_github   else 0
    ats += min(12, bullet_count * 1.5)  # bullet richness
    ats += 8  if has_metrics  else 0
    ats += min(10, strong_count * 2)    # strong verbs
    ats -= min(10, weak_count   * 3)    # weak verbs penalty
    ats -= min(8,  buzz_count   * 2)    # buzzword penalty
    ats_score = int(min(95, max(20, ats)))

    # Formatting (section completeness)
    fmt = 20
    fmt += 15 if has_exp      else 0
    fmt += 15 if has_edu      else 0
    fmt += 12 if has_skills   else 0
    fmt += 10 if has_summary  else 0
    fmt += 8  if has_projects else 0
    fmt += 6  if has_certs    else 0
    fmt += 7  if has_linkedin else 0
    fmt += 7  if has_github   else 0
    formatting_score = int(min(95, max(20, fmt)))

    # Content quality
    content = 20
    content += min(25, digit_count // 2)   # quantification
    content += min(20, strong_count * 3)   # action verbs
    content += min(15, bullet_count * 2)   # bullet count
    content -= min(15, weak_count * 4)     # weak language
    content_score = int(min(95, max(20, content)))

    # Skills
    skills_score = int(min(95, max(15, 15 + skill_count * 6)))

    # Keywords (density-based)
    density = skill_count / max(len(words), 1) * 100
    keywords_score = int(min(95, max(15, density * 10 + 20)))

    # Readability
    sentences = [s for s in re.split(r'[.!\n]', resume_text) if len(s.strip()) > 8]
    avg_len   = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else 20
    readability = int(min(95, max(30, 90 - max(0, avg_len - 15) * 1.5 - buzz_count * 5)))

    return {
        'ats_score':         ats_score,
        'formatting_score':  formatting_score,
        'content_score':     content_score,
        'skills_score':      skills_score,
        'keywords_score':    keywords_score,
        'readability_score': readability,
    }


def full_analysis_lite(resume_text: str, job_description: str = '') -> dict:
    """Complete analysis without sklearn models."""
    scores      = predict_scores_lite(resume_text)
    keywords    = extract_keywords(resume_text)
    weakness    = detect_weaknesses(resume_text)

    overall = (
        scores['ats_score']         * 0.30 +
        scores['formatting_score']  * 0.15 +
        scores['content_score']     * 0.25 +
        scores['skills_score']      * 0.15 +
        scores['keywords_score']    * 0.10 +
        scores['readability_score'] * 0.05
    )

    meta   = _load_meta()
    STRONG = meta['strong_verbs']
    WEAK   = meta['weak_verbs']
    tl     = resume_text.lower()

    suggestions = []
    if scores['ats_score'] < 70:
        suggestions.append('Add industry keywords from job descriptions to boost ATS score')
    if scores['content_score'] < 65:
        suggestions.append('Quantify achievements: add numbers, percentages, and $ impact')
    if scores['formatting_score'] < 70:
        suggestions.append('Add missing sections: professional summary, projects, certifications')
    if weakness['issues_detail'].get('weak_action_verbs'):
        suggestions.append("Replace weak phrases like 'Responsible for' with Led, Built, Engineered")
    if weakness['issues_detail'].get('missing_github'):
        suggestions.append('Add GitHub profile URL to showcase your technical work')
    suggestions.append('Tailor your resume summary to mirror each job description you apply to')

    return {
        **scores,
        'found_keywords':   keywords['found_keywords'],
        'missing_keywords': keywords['missing_keywords'],
        'strengths':        weakness['strengths'][:4],
        'weaknesses':       weakness['weaknesses'][:4],
        'suggestions':      suggestions[:5],
        'grammar_issues':   [],
        'action_verbs': {
            'strong': [v for v in STRONG if v.lower() in tl][:5],
            'weak':   [v for v in WEAK   if v.lower() in tl][:3],
        },
        'overall_feedback': (
            f'Your resume scores {int(overall)}/100 overall. '
            f"{'Strong ATS compatibility.' if scores['ats_score'] >= 70 else 'Needs ATS keyword optimization.'} "
            f"{'Good content with quantified achievements.' if scores['content_score'] >= 65 else 'Add more measurable achievements.'} "
            f"Priority: {', '.join(keywords['missing_keywords'][:3]) if keywords['missing_keywords'] else 'maintain current quality'}."
        ),
    }