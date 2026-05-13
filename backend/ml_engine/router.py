"""
ResumeAI Pro — AI Router
Priority order:
  1. Gemini API      → if GEMINI_API_KEY set in env
  2. ML Engine       → if .pkl models trained locally
  3. Rule-based Lite → always works, no dependencies (Vercel default)
"""
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def _use_gemini() -> bool:
    key = getattr(settings, "GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    return bool(key and key not in ("", "your-gemini-api-key", "REPLACE_ME"))


def _ml_available() -> bool:
    try:
        from ml_engine.inference import ModelStore
        return ModelStore.available()
    except Exception:
        return False


def get_backend() -> str:
    if _use_gemini():
        return "gemini"
    if _ml_available():
        return "ml"
    return "lite"           # rule-based, always works


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_resume(resume_text: str, job_description: str = "") -> dict:
    b = get_backend()
    logger.info(f"analyze_resume → {b}")
    if b == "gemini":
        return _gemini_analyze(resume_text, job_description)
    if b == "ml":
        from ml_engine.inference import full_analysis
        return full_analysis(resume_text, job_description)
    from ml_engine.inference_lite import full_analysis_lite
    return full_analysis_lite(resume_text, job_description)


def ats_match(resume_text: str, job_description: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_ats_match(resume_text, job_description)
    from ml_engine.inference import compute_ats_match
    return compute_ats_match(resume_text, job_description)


def skill_gap(resume_text: str, job_description: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_skill_gap(resume_text, job_description)
    from ml_engine.inference import analyze_skill_gap
    return analyze_skill_gap(resume_text, job_description)


def improve_resume(resume_text: str, target_role: str = "") -> str:
    b = get_backend()
    if b == "gemini":
        return _gemini_improve(resume_text, target_role)
    from ml_engine.router_helpers import ml_improve_resume
    return ml_improve_resume(resume_text)


def enhance_bullet(bullet: str, context: str = "") -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_enhance_bullet(bullet, context)
    from ml_engine.inference import score_bullet
    return score_bullet(bullet)


def detect_weaknesses_ai(resume_text: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_weaknesses(resume_text)
    from ml_engine.inference import detect_weaknesses
    return detect_weaknesses(resume_text)


def check_grammar_ai(resume_text: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_grammar(resume_text)
    from ml_engine.inference import check_grammar
    return check_grammar(resume_text)


def generate_cover_letter(resume_text: str, company: str, role: str,
                          tone: str = "professional", job_description: str = "") -> str:
    b = get_backend()
    if b == "gemini":
        return _gemini_cover_letter(resume_text, company, role, tone, job_description)
    from ml_engine.inference import generate_cover_letter_ml
    return generate_cover_letter_ml(resume_text, company, role, tone, job_description)


def recommend_roles(resume_text: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_roles(resume_text)
    from ml_engine.inference import predict_job_roles
    return predict_job_roles(resume_text)


def job_apply_analysis(resume_text: str, job_title: str, job_description: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_job_apply(resume_text, job_title, job_description)
    from ml_engine.router_helpers import ml_job_apply
    return ml_job_apply(resume_text, job_title, job_description)


def detect_fake(resume_text: str) -> dict:
    b = get_backend()
    if b == "gemini":
        return _gemini_fake(resume_text)
    from ml_engine.router_helpers import ml_fake_detect
    return ml_fake_detect(resume_text)


def generate_portfolio_html(resume_text: str, theme: str = "dark") -> str:
    b = get_backend()
    if b == "gemini":
        return _gemini_portfolio(resume_text, theme)
    from ml_engine.router_helpers import ml_portfolio
    return ml_portfolio(resume_text, theme)


def chat_reply(message: str, history: list) -> str:
    b = get_backend()
    if b == "gemini":
        return _gemini_chat(message, history)
    from ml_engine.router_helpers import ml_chat
    return ml_chat(message)


# ── Gemini implementations ────────────────────────────────────────────────────

def _g(prompt, max_tokens=2048):
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
    )
    return resp.text.strip()


def _gj(prompt):
    import json
    raw   = _g(prompt + "\n\nRespond ONLY with valid JSON. No markdown.")
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def _gemini_analyze(rt, jd):
    return _gj(f"""ATS expert. Analyze resume. Return JSON:
{{"ats_score":0-100,"formatting_score":0-100,"content_score":0-100,"skills_score":0-100,
"keywords_score":0-100,"readability_score":0-100,"found_keywords":[],"missing_keywords":[],
"strengths":[],"weaknesses":[],"suggestions":[],"grammar_issues":[],"action_verbs":{{"strong":[],"weak":[]}},"overall_feedback":""}}
Resume:{rt}
{f"JD:{jd}" if jd else ""}""")

def _gemini_ats_match(rt, jd):
    return _gj(f"ATS match. Return JSON:match_score,keyword_match_percentage,matched_keywords,missing_critical_keywords,missing_nice_to_have,recommendation,reasoning,optimization_tips\nResume:{rt}\nJD:{jd}")

def _gemini_skill_gap(rt, jd):
    return _gj(f"Skill gap. Return JSON:matching_skills,critical_missing_skills,nice_to_have_missing,transferable_skills,learning_path,overall_readiness,summary\nResume:{rt}\nJD:{jd}")

def _gemini_improve(rt, role):
    return _g(f"Rewrite resume professionally with strong action verbs, ATS keywords, quantified achievements.{f' Target role:{role}' if role else ''}\nResume:{rt}", 2000)

def _gemini_enhance_bullet(b, ctx):
    return _gj(f"Improve bullet. Return JSON:alternatives(3 with text+reason),issues_with_original,techniques_used\nBullet:{b}")

def _gemini_weaknesses(rt):
    return _gj(f"Detect weaknesses. Return JSON:overall_weakness_score,weaknesses,strengths,issues_detail\nResume:{rt}")

def _gemini_grammar(rt):
    return _gj(f"Grammar check. Return JSON:grammar_errors,passive_voice_instances,long_sentences,readability_score,reading_level,consistency_issues,formatting_issues,overall_grade,summary\nResume:{rt}")

def _gemini_cover_letter(rt, co, ro, tone, jd):
    return _g(f"Write {tone} cover letter for {ro} at {co}.\nResume:{rt}\n{f'JD:{jd}' if jd else ''}\nReturn only letter text.", 800)

def _gemini_roles(rt):
    return _gj(f"Job roles. Return JSON:recommendations(title,match_percentage,why_good_fit,salary_range,required_upskilling,top_companies_hiring),career_trajectory,strongest_industries\nResume:{rt}")

def _gemini_job_apply(rt, jt, jd):
    return _gj(f"Should apply for '{jt}'? Return JSON:match_score,verdict,verdict_color,top_strengths,key_gaps,application_tips,interview_talking_points,resume_tweaks_needed,likelihood_of_callback,reasoning\nResume:{rt}\nJD:{jd}")

def _gemini_fake(rt):
    return _gj(f"Fake resume detect. Return JSON:credibility_score,risk_level,red_flags,exaggerated_claims,overused_buzzwords,vague_statements,timeline_inconsistencies,suspicious_achievements,authenticity_indicators,recommendations_for_candidate,overall_assessment\nResume:{rt}")

def _gemini_portfolio(rt, theme):
    html = _g(f"Generate complete single-file HTML portfolio. Theme:{theme}.\nResume:{rt}\nReturn ONLY raw HTML.", 4096)
    start = html.find("<!DOCTYPE")
    return html[start:] if start >= 0 else html

def _gemini_chat(message, history):
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="You are an expert career coach. Help with resumes, job search, interviews, salary negotiation. Be concise and actionable."
    )
    chat = model.start_chat(history=history)
    return chat.send_message(message).text.strip()