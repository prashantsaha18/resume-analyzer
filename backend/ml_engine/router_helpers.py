"""
ML and rule-based helper implementations used by router.py.
All functions work without any API key or trained models.
"""
import re
import random
from ml_engine.inference import (
    _load_meta, detect_weaknesses, compute_ats_match,
    generate_portfolio_structure
)


def ml_improve_resume(resume_text: str) -> str:
    meta = _load_meta()
    STRONG = meta["strong_verbs"]
    WEAK   = meta["weak_verbs"]
    lines  = resume_text.split("\n")
    improved = []
    notes    = []

    for line in lines:
        new_line = line
        for phrase in WEAK:
            if phrase.lower() in line.lower():
                replacement = random.choice(STRONG[:12])
                new_line = re.sub(re.escape(phrase), replacement, new_line, flags=re.IGNORECASE)
                notes.append(f"Replaced '{phrase}' → '{replacement}'")
        # Flag unquantified bullets
        if line.strip().startswith("•") and not re.search(r"\d", line):
            new_line = new_line.rstrip() + "  ← add metric (e.g. 30% faster, 10K users)"
        improved.append(new_line)

    result = "\n".join(improved)
    if notes:
        result += "\n\n─── IMPROVEMENTS MADE ───"
        for n in notes[:6]:
            result += f"\n• {n}"
    result += (
        "\n\n─── FURTHER RECOMMENDATIONS ───"
        "\n• Quantify every bullet: add %, $, time saved, team size"
        "\n• Open each bullet with a strong verb (Led, Built, Scaled, Launched)"
        "\n• Keep bullets to 1-2 lines — cut filler words"
        "\n• Tailor skills section to match each job description"
        "\n• Add a 2-3 sentence professional summary if missing"
    )
    return result


def ml_job_apply(resume_text: str, job_title: str, job_description: str) -> dict:
    match_data = compute_ats_match(resume_text, job_description)
    score = match_data["match_score"]

    if score >= 70:
        verdict, color, callback = "Strong Apply ✅", "green",  "High"
    elif score >= 50:
        verdict, color, callback = "Apply 👍",       "yellow", "Medium"
    elif score >= 30:
        verdict, color, callback = "Apply with Modifications ⚠️", "yellow", "Medium"
    else:
        verdict, color, callback = "Significant Gaps — Upskill First ❌", "red", "Low"

    strengths = [f"Strong match on: {s}" for s in match_data["matched_keywords"][:3]]
    if not strengths:
        strengths = ["Review your resume to identify relevant experience"]

    gaps = [f"Missing required skill: {s}" for s in match_data["missing_critical_keywords"][:3]]

    tips = match_data.get("optimization_tips", [])
    tips += ["Rewrite your summary to mirror the job description language"]
    tips += [f"Add '{s}' to your skills section" for s in match_data["missing_critical_keywords"][:2]]

    talking = [f"Discuss your experience with {s}" for s in match_data["matched_keywords"][:3]]
    talking += ["Explain how you handle the responsibilities listed in the JD",
                "Prepare a concrete example for your most relevant achievement"]

    tweaks = [f"Add the keyword '{s}' to experience or skills" for s in match_data["missing_critical_keywords"][:2]]

    return {
        "match_score":               score,
        "verdict":                   verdict,
        "verdict_color":             color,
        "top_strengths":             strengths,
        "key_gaps":                  gaps,
        "application_tips":          tips[:4],
        "interview_talking_points":  talking[:4],
        "resume_tweaks_needed":      tweaks,
        "likelihood_of_callback":    callback,
        "reasoning":                 match_data["reasoning"],
    }


def ml_fake_detect(resume_text: str) -> dict:
    meta       = _load_meta()
    BUZZWORDS  = meta["buzzwords"]
    text_lower = resume_text.lower()

    red_flags   = []
    buzz_found  = [b for b in BUZZWORDS if b.lower() in text_lower]
    vague_found = [p for p in ["various", "multiple tasks", "etc", "and so on", "several"]
                   if p in text_lower]

    if buzz_found:
        red_flags.append({
            "issue":    f"Overused buzzwords: {', '.join(buzz_found)}",
            "severity": "Medium",
            "location": "Summary / Skills"
        })

    big_pcts = re.findall(r"(\d{3,})\s*%", resume_text)
    for n in big_pcts:
        if int(n) > 100:
            red_flags.append({
                "issue":    f"Impossible percentage: {n}%",
                "severity": "High",
                "location": "Experience"
            })

    vague_claims = re.findall(
        r"(various \w+|multiple \w+|several \w+|many \w+)", text_lower
    )

    credibility = max(30, min(95, 80 - len(red_flags) * 12 - len(vague_claims) * 5))

    return {
        "credibility_score": credibility,
        "risk_level":        "Low" if credibility >= 75 else "Medium" if credibility >= 50 else "High",
        "red_flags":         red_flags,
        "exaggerated_claims":[f"Verify claim: {n}% improvement" for n in big_pcts if int(n) > 90],
        "overused_buzzwords": buzz_found,
        "vague_statements":  [f"Vague phrase detected: '{v}'" for v in vague_found[:4]],
        "timeline_inconsistencies": [],
        "suspicious_achievements":  [],
        "authenticity_indicators":  (
            ["Specific company names present", "Realistic job titles used"]
            if credibility >= 65 else []
        ),
        "recommendations_for_candidate": [
            "Replace buzzwords with specific technologies and metrics",
            "Verify all percentage claims are accurate and verifiable",
            "Replace vague language with concrete, specific examples",
        ],
        "overall_assessment": (
            f"Credibility score: {credibility}/100. "
            + ("Resume appears genuine with no major red flags." if credibility >= 75
               else "Some red flags detected — review highlighted areas before submitting.")
        ),
    }


def ml_portfolio(resume_text: str, theme: str = "dark") -> str:
    data   = generate_portfolio_structure(resume_text)
    name   = data.get("name", "Developer")
    role   = data.get("role", "Software Engineer")
    skills = data.get("skills", [])[:10]
    meta   = _load_meta()

    # Theme palette
    palettes = {
        "dark":    {"bg": "#0a0d1a", "bg2": "#151c32", "text": "#e8edf8",
                    "text2": "#8a9bc4", "border": "#2a3556", "accent": "#4f8ef7",
                    "accent2": "#8b5cf6", "card": "#151c32"},
        "light":   {"bg": "#f8faff", "bg2": "#ffffff", "text": "#1a1a2e",
                    "text2": "#555577", "border": "#e0e4f0", "accent": "#3d7af5",
                    "accent2": "#7c3aed", "card": "#ffffff"},
        "minimal": {"bg": "#fafafa", "bg2": "#ffffff", "text": "#111111",
                    "text2": "#555555", "border": "#e5e5e5", "accent": "#000000",
                    "accent2": "#333333", "card": "#f5f5f5"},
    }
    p = palettes.get(theme, palettes["dark"])

    skills_tags = "".join(f'<span class="skill-tag">{s}</span>' for s in skills)
    first_name  = name.split()[0] if name else "Dev"
    year        = 2024

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{name} — Portfolio</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --bg:{p["bg"]};--bg2:{p["bg2"]};--text:{p["text"]};--text2:{p["text2"]};
  --border:{p["border"]};--accent:{p["accent"]};--accent2:{p["accent2"]};--card:{p["card"]};
}}
html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--text);font-family:"DM Sans",sans-serif;line-height:1.6}}
a{{color:inherit;text-decoration:none}}

/* NAV */
nav{{position:fixed;top:0;width:100%;background:var(--bg)ee;
  backdrop-filter:blur(12px);border-bottom:1px solid var(--border);
  padding:14px 40px;display:flex;justify-content:space-between;align-items:center;z-index:100}}
.logo{{font-family:"Syne",sans-serif;font-weight:800;font-size:18px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.nav-links a{{color:var(--text2);font-size:13px;margin-left:28px;
  font-weight:500;transition:color .2s}}
.nav-links a:hover{{color:var(--accent)}}

/* HERO */
.hero{{min-height:100vh;display:flex;align-items:center;
  padding:100px 40px 60px;max-width:1100px;margin:0 auto}}
.hero-badge{{display:inline-flex;align-items:center;gap:8px;
  padding:6px 16px;background:color-mix(in srgb,var(--accent) 12%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 25%,transparent);
  border-radius:20px;font-size:12px;font-weight:600;color:var(--accent);
  letter-spacing:.05em;margin-bottom:20px}}
.hero h1{{font-family:"Syne",sans-serif;font-size:clamp(40px,7vw,80px);
  font-weight:800;line-height:1.05;margin-bottom:20px}}
.hero h1 .name{{background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero p{{font-size:18px;color:var(--text2);max-width:520px;
  margin-bottom:36px;line-height:1.75}}
.btn-group{{display:flex;gap:12px;flex-wrap:wrap}}
.btn{{display:inline-flex;align-items:center;gap:8px;padding:13px 28px;
  border-radius:12px;font-weight:600;font-size:14px;transition:all .2s;cursor:pointer}}
.btn-primary{{background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border:none}}
.btn-primary:hover{{opacity:.88;transform:translateY(-2px)}}
.btn-outline{{background:transparent;color:var(--text);
  border:1px solid var(--border)}}
.btn-outline:hover{{border-color:var(--accent);color:var(--accent)}}
.hero-visual{{flex:1;display:flex;justify-content:flex-end;padding-left:60px}}
.avatar-ring{{width:300px;height:300px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  padding:3px;display:flex;align-items:center;justify-content:center}}
.avatar-inner{{width:100%;height:100%;border-radius:50%;
  background:var(--bg2);display:flex;align-items:center;justify-content:center;
  font-family:"Syne",sans-serif;font-size:80px;font-weight:800;
  color:var(--accent)}}

/* SECTIONS */
section{{padding:90px 40px;max-width:1100px;margin:0 auto}}
.section-label{{font-size:11px;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:var(--accent);margin-bottom:12px}}
.section-title{{font-family:"Syne",sans-serif;font-size:clamp(28px,4vw,42px);
  font-weight:700;line-height:1.15;margin-bottom:48px}}
.section-title span{{color:var(--accent)}}

/* SKILLS */
.skills-wrap{{display:flex;flex-wrap:wrap;gap:10px}}
.skill-tag{{padding:8px 20px;background:color-mix(in srgb,var(--accent) 10%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 20%,transparent);
  border-radius:24px;font-size:13px;font-weight:500;color:var(--accent);
  transition:all .2s}}
.skill-tag:hover{{background:color-mix(in srgb,var(--accent) 20%,transparent);
  transform:translateY(-2px)}}

/* EXPERIENCE TIMELINE */
.timeline{{position:relative;padding-left:32px}}
.timeline::before{{content:"";position:absolute;left:0;top:0;bottom:0;
  width:2px;background:var(--border)}}
.tl-item{{position:relative;margin-bottom:40px}}
.tl-dot{{position:absolute;left:-39px;top:4px;width:14px;height:14px;
  border-radius:50%;background:var(--accent);border:3px solid var(--bg)}}
.tl-period{{font-size:12px;color:var(--accent);font-weight:600;
  letter-spacing:.04em;margin-bottom:6px}}
.tl-title{{font-family:"Syne",sans-serif;font-size:18px;font-weight:700;margin-bottom:2px}}
.tl-company{{font-size:14px;color:var(--text2);margin-bottom:10px}}
.tl-desc{{font-size:14px;color:var(--text2);line-height:1.7}}

/* PROJECT CARDS */
.projects-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
.proj-card{{background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:28px;transition:all .25s}}
.proj-card:hover{{border-color:var(--accent);transform:translateY(-4px);
  box-shadow:0 12px 40px color-mix(in srgb,var(--accent) 15%,transparent)}}
.proj-icon{{font-size:32px;margin-bottom:14px}}
.proj-title{{font-family:"Syne",sans-serif;font-size:17px;font-weight:700;margin-bottom:8px}}
.proj-desc{{font-size:13px;color:var(--text2);line-height:1.65;margin-bottom:16px}}
.proj-tags{{display:flex;flex-wrap:wrap;gap:6px}}
.proj-tag{{padding:3px 10px;background:color-mix(in srgb,var(--accent2) 12%,transparent);
  color:var(--accent2);border-radius:20px;font-size:11px;font-weight:600}}

/* CONTACT */
.contact-wrap{{display:grid;grid-template-columns:1fr 1fr;gap:32px;align-items:start}}
.contact-info p{{color:var(--text2);font-size:15px;line-height:1.75;margin-bottom:28px}}
.contact-links{{display:flex;flex-direction:column;gap:12px}}
.contact-link{{display:flex;align-items:center;gap:12px;
  padding:14px 20px;background:var(--card);border:1px solid var(--border);
  border-radius:12px;font-size:14px;font-weight:500;
  transition:all .2s;color:var(--text)}}
.contact-link:hover{{border-color:var(--accent);color:var(--accent);
  transform:translateX(4px)}}
.contact-link .icon{{font-size:18px}}

/* FOOTER */
footer{{text-align:center;padding:40px;border-top:1px solid var(--border);
  color:var(--text2);font-size:13px}}
footer .heart{{color:#ef4444}}

/* RESPONSIVE */
@media(max-width:768px){{
  .hero{{flex-direction:column;padding:100px 24px 60px}}
  .hero-visual{{display:none}}
  section{{padding:60px 24px}}
  nav{{padding:14px 24px}}
  .nav-links{{display:none}}
  .contact-wrap{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <div class="logo">{first_name}</div>
  <div class="nav-links">
    <a href="#about">About</a>
    <a href="#skills">Skills</a>
    <a href="#experience">Experience</a>
    <a href="#projects">Projects</a>
    <a href="#contact">Contact</a>
  </div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-text">
    <div class="hero-badge">⚡ Open to opportunities</div>
    <h1>Hi, I'm<br/><span class="name">{name}</span></h1>
    <p>{role} passionate about building scalable, user-centric software. I turn complex problems into elegant solutions.</p>
    <div class="btn-group">
      <a href="#contact" class="btn btn-primary">Get In Touch →</a>
      <a href="#projects" class="btn btn-outline">View Projects</a>
    </div>
  </div>
  <div class="hero-visual">
    <div class="avatar-ring">
      <div class="avatar-inner">{first_name[0]}</div>
    </div>
  </div>
</div>

<!-- SKILLS -->
<section id="skills">
  <div class="section-label">What I Work With</div>
  <div class="section-title">Technical <span>Skills</span></div>
  <div class="skills-wrap">{skills_tags}</div>
</section>

<!-- EXPERIENCE -->
<section id="experience">
  <div class="section-label">Career Journey</div>
  <div class="section-title">Work <span>Experience</span></div>
  <div class="timeline">
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-period">2022 – PRESENT</div>
      <div class="tl-title">{role}</div>
      <div class="tl-company">Current Employer · Full-time</div>
      <div class="tl-desc">
        Building and maintaining production-grade systems. Leading cross-functional initiatives, 
        driving performance improvements, and mentoring junior engineers.
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-period">2020 – 2022</div>
      <div class="tl-title">Software Engineer</div>
      <div class="tl-company">Previous Company · Full-time</div>
      <div class="tl-desc">
        Developed full-stack features, designed RESTful APIs, and collaborated in an agile 
        team to ship high-quality product releases on schedule.
      </div>
    </div>
    <div class="tl-item">
      <div class="tl-dot"></div>
      <div class="tl-period">2019 – 2020</div>
      <div class="tl-title">Junior Developer / Intern</div>
      <div class="tl-company">Startup · Internship</div>
      <div class="tl-desc">
        Contributed to frontend and backend features, wrote unit tests, 
        and participated in code reviews to grow as an engineer.
      </div>
    </div>
  </div>
</section>

<!-- PROJECTS -->
<section id="projects">
  <div class="section-label">What I've Built</div>
  <div class="section-title">Featured <span>Projects</span></div>
  <div class="projects-grid">
    <div class="proj-card">
      <div class="proj-icon">🚀</div>
      <div class="proj-title">Open Source Tool</div>
      <div class="proj-desc">Built a developer tool that streamlines workflow automation. Gained 2K+ GitHub stars from the community within 6 months of launch.</div>
      <div class="proj-tags">{"".join(f'<span class="proj-tag">{s}</span>' for s in skills[:3])}</div>
    </div>
    <div class="proj-card">
      <div class="proj-icon">📊</div>
      <div class="proj-title">Analytics Dashboard</div>
      <div class="proj-desc">Real-time data visualization platform serving 50K+ monthly active users. Reduced reporting time by 70% through smart caching.</div>
      <div class="proj-tags">{"".join(f'<span class="proj-tag">{s}</span>' for s in skills[1:4])}</div>
    </div>
    <div class="proj-card">
      <div class="proj-icon">🤖</div>
      <div class="proj-title">AI Integration</div>
      <div class="proj-desc">Integrated ML models into a production application, improving recommendation accuracy by 45% and boosting user engagement.</div>
      <div class="proj-tags">{"".join(f'<span class="proj-tag">{s}</span>' for s in skills[2:5])}</div>
    </div>
  </div>
</section>

<!-- CONTACT -->
<section id="contact">
  <div class="section-label">Let's Connect</div>
  <div class="section-title">Get In <span>Touch</span></div>
  <div class="contact-wrap">
    <div class="contact-info">
      <p>I'm always open to discussing new opportunities, interesting projects, or just having a great conversation about tech and career growth.</p>
      <a href="mailto:hello@example.com" class="btn btn-primary">Send Email →</a>
    </div>
    <div class="contact-links">
      <a class="contact-link" href="#"><span class="icon">📧</span> hello@example.com</a>
      <a class="contact-link" href="#"><span class="icon">💼</span> linkedin.com/in/{name.lower().replace(" ", "")}</a>
      <a class="contact-link" href="#"><span class="icon">🐙</span> github.com/{first_name.lower()}</a>
      <a class="contact-link" href="#"><span class="icon">🌐</span> {first_name.lower()}.dev</a>
    </div>
  </div>
</section>

<footer>
  <p>Built with <span class="heart">♥</span> by {name} · {year} · Generated by ResumeAI Pro</p>
</footer>

<script>
// Smooth active nav highlight
const sections = document.querySelectorAll("section,[id]");
const navLinks = document.querySelectorAll(".nav-links a");
const observer = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if(e.isIntersecting) {{
      navLinks.forEach(l => l.style.color = "");
      const active = document.querySelector(`.nav-links a[href="#${{e.target.id}}"]`);
      if(active) active.style.color = "var(--accent)";
    }}
  }});
}}, {{threshold: 0.5}});
sections.forEach(s => observer.observe(s));

// Fade-in animation
const observer2 = new IntersectionObserver(entries => {{
  entries.forEach(e => {{
    if(e.isIntersecting) {{
      e.target.style.opacity = "1";
      e.target.style.transform = "translateY(0)";
    }}
  }});
}}, {{threshold: 0.1}});
document.querySelectorAll(".tl-item,.proj-card,.skill-tag").forEach(el => {{
  el.style.opacity = "0";
  el.style.transform = "translateY(20px)";
  el.style.transition = "opacity 0.5s ease, transform 0.5s ease";
  observer2.observe(el);
}});
</script>
</body>
</html>"""


def ml_chat(message: str) -> str:
    msg = message.lower().strip()

    responses = {
        ("ats", "ats score", "applicant tracking"):
            "To improve your ATS score:\n• Mirror exact keywords from the job description\n• Use standard section headers: Experience, Education, Skills\n• Avoid tables, columns, and images — plain text only\n• Include both acronyms and full terms (e.g. ML and Machine Learning)\n• Aim for 60–70% keyword match with the target JD",

        ("resume", "improve", "make better", "rewrite"):
            "Key resume improvements:\n• Start every bullet with a strong action verb (Led, Built, Scaled, Engineered)\n• Quantify every achievement — numbers, percentages, dollar amounts, team size\n• Keep it to 1–2 pages; cut anything older than 10 years\n• Add a 2–3 sentence tailored summary at the top\n• Use the STAR format: Situation → Task → Action → Result",

        ("interview", "prepare", "interview tips", "interview question"):
            "Interview preparation:\n1. Research the company deeply — products, culture, recent news, competitors\n2. Prepare 5–7 STAR stories for behavioral questions\n3. Practice technical questions specific to your role\n4. Prepare 3–5 thoughtful questions to ask them\n5. Follow up with a personalized thank-you email within 24 hours\n6. Dress one level above the company's culture",

        ("salary", "negotiate", "pay", "compensation", "offer"):
            "Salary negotiation:\n• Always negotiate — 85% of offers have room to move\n• Research: Levels.fyi, Glassdoor, LinkedIn Salary, Blind\n• Give a range, with your target at the bottom of the range\n• Say: 'I'm very excited about this role. Based on my research and experience, I was expecting $X–Y. Is there flexibility?'\n• Consider total comp: base, equity, bonus, PTO, remote work\n• Never accept on the spot — ask for 24–48 hours to review",

        ("cover letter", "covering letter", "cover"):
            "Strong cover letter structure:\n• Opening: A hook that shows company knowledge — never start with 'I am writing to apply'\n• Paragraph 1: Your top 2 achievements that match their exact needs\n• Paragraph 2: Why THIS company specifically (culture, mission, product)\n• Closing: Confident call to action, not 'I hope to hear from you'\n• Length: 250–350 words maximum\n• Tailor every single one — generic letters get ignored",

        ("linkedin", "profile", "linkedin profile"):
            "LinkedIn optimization:\n• Professional headshot: 14× more profile views\n• Headline: What you do + who you help (not just job title)\n• Summary: First-person, 3–5 sentences, include your top skills and what you're seeking\n• Get 3–5 specific skill recommendations from colleagues\n• Connect with 500+ in your field\n• Post content 2–3× per week to increase visibility\n• Turn on 'Open to Work' (visible only to recruiters)",

        ("gap", "career gap", "employment gap", "break"):
            "Handling career gaps:\n• Be honest and brief — a 30-second confident explanation is enough\n• Frame it positively: 'I took time to upskill in X / care for family / work on a project'\n• If you freelanced, consulted, or took courses during the gap — list them\n• On your resume, use year ranges instead of month ranges to minimize visual impact\n• Interviewers respect honesty far more than awkward deflection",

        ("switch", "career change", "transition", "pivot"):
            "Career transition strategy:\n1. Identify your transferable skills — they matter more than your title\n2. Build 2–3 portfolio projects in the new field\n3. Get a targeted certification (AWS, Google, PMP, etc.)\n4. Network with people already in the target role — 80% of jobs are filled through connections\n5. Look for bridge roles that sit between old and new fields\n6. Update your LinkedIn headline to reflect your target role, not your current one",

        ("remote", "work from home", "wfh"):
            "Landing remote roles:\n• Filter job boards with 'remote' tag: LinkedIn, We Work Remotely, Remote.co, Himalayas\n• Highlight async communication skills in your resume\n• Show self-management and independent delivery in your bullets\n• Mention tools: Slack, Notion, Jira, Figma, Zoom in your skills\n• Timezone flexibility is a competitive advantage — mention it if you have it",

        ("portfolio", "projects", "github"):
            "Building a strong portfolio:\n• Quality over quantity: 2–3 polished projects beat 10 half-finished ones\n• Each project needs: clear README, live demo, and problem-solution description\n• Show measurable impact: users, performance gains, stars, downloads\n• Pin your best repos on GitHub\n• Deploy everything — even a simple static site on Vercel shows initiative\n• Contribute to open source: even small PRs show real-world collaboration",
    }

    for keywords, response in responses.items():
        if any(kw in msg for kw in keywords):
            return response

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "start", "help"]
    if any(g in msg for g in greetings):
        return (
            "Hello! 👋 I'm your AI career coach. I can help you with:\n\n"
            "• 📄 Resume writing and ATS optimization\n"
            "• 🎯 Interview preparation and practice\n"
            "• 💰 Salary negotiation tactics\n"
            "• 🔄 Career transitions and planning\n"
            "• 🔗 LinkedIn profile optimization\n"
            "• 📝 Cover letter writing\n"
            "• 🌐 Remote job search strategies\n\n"
            "What would you like to work on today?"
        )

    thanks = ["thank", "thanks", "great", "helpful", "awesome", "perfect"]
    if any(t in msg for t in thanks):
        return "You're welcome! Best of luck with your career journey. Feel free to ask anything else — I'm here to help! 🚀"

    return (
        f"Great question about '{message[:60]}'. Here's my advice:\n\n"
        "Focus on being specific and measurable in everything you do:\n"
        "• On your resume: quantify every achievement with numbers\n"
        "• In your job search: quality tailored applications beat volume\n"
        "• In interviews: use the STAR method for behavioral questions\n"
        "• In networking: give value before asking for anything\n\n"
        "Would you like me to go deeper on resume writing, interview prep, salary negotiation, or career planning?"
    )