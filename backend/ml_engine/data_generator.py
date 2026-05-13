"""
Synthetic Resume Data Generator
Generates 5000+ labeled resume samples for ML training.
Each sample has a resume text + ground truth scores for all dimensions.
"""

import random
import json
import numpy as np

# ─── Vocabulary Banks ──────────────────────────────────────────────────────────

STRONG_VERBS = [
    "Led", "Built", "Engineered", "Architected", "Spearheaded", "Developed",
    "Designed", "Implemented", "Optimized", "Scaled", "Launched", "Delivered",
    "Transformed", "Automated", "Reduced", "Increased", "Improved", "Managed",
    "Mentored", "Collaborated", "Deployed", "Migrated", "Refactored", "Streamlined",
    "Established", "Pioneered", "Orchestrated", "Accelerated", "Modernized", "Drove",
    "Executed", "Generated", "Achieved", "Exceeded", "Secured", "Negotiated",
    "Analyzed", "Resolved", "Created", "Integrated", "Maintained", "Oversaw",
]

WEAK_VERBS = [
    "Responsible for", "Helped with", "Worked on", "Was involved in",
    "Assisted with", "Participated in", "Did", "Made", "Tried to",
    "Contributed to", "Part of the team that", "Helped", "Was part of",
]

TECH_SKILLS = [
    "Python", "JavaScript", "TypeScript", "React", "Node.js", "Django",
    "FastAPI", "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes",
    "AWS", "GCP", "Azure", "Git", "CI/CD", "GraphQL", "REST API",
    "Machine Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy",
    "Java", "Spring Boot", "Go", "Rust", "C++", "C#", ".NET",
    "Vue.js", "Angular", "Next.js", "Express.js", "Flask",
    "MySQL", "Elasticsearch", "RabbitMQ", "Kafka", "Terraform",
    "Linux", "Bash", "Nginx", "Jenkins", "GitHub Actions",
    "Microservices", "System Design", "Data Structures", "Algorithms",
]

SOFT_SKILLS = [
    "Leadership", "Communication", "Problem Solving", "Team Collaboration",
    "Critical Thinking", "Time Management", "Adaptability", "Mentoring",
    "Agile", "Scrum", "Cross-functional collaboration", "Stakeholder management",
]

COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Apple", "Netflix", "Uber",
    "Airbnb", "Stripe", "Shopify", "Salesforce", "Oracle", "IBM",
    "Accenture", "Deloitte", "TechCorp", "StartupXYZ", "InnovateTech",
    "DataSystems Inc", "CloudBase", "DevStudio", "FinTech Solutions",
    "HealthTech Co", "EduPlatform", "RetailTech", "CyberSecure",
]

UNIVERSITIES = [
    "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon",
    "University of Michigan", "Georgia Tech", "Purdue University",
    "University of Texas", "Ohio State University", "Penn State",
    "State University", "City College", "Tech Institute",
    "National University", "Regional College",
]

DEGREES = [
    "B.S. Computer Science", "B.Tech Computer Engineering",
    "M.S. Software Engineering", "B.S. Information Technology",
    "M.S. Computer Science", "B.S. Data Science",
    "M.S. Machine Learning", "B.E. Electronics & CS",
]

CERTS = [
    "AWS Certified Developer", "AWS Solutions Architect",
    "Google Cloud Professional", "CKA Kubernetes Administrator",
    "PMP Certified", "Scrum Master (CSM)", "Azure Developer Associate",
    "Oracle Java Certified", "Cisco CCNA", "CompTIA Security+",
]

BUZZWORDS = [
    "synergy", "leverage", "paradigm shift", "disruptive", "innovative",
    "thought leader", "guru", "ninja", "rockstar", "wizard",
    "holistic approach", "ecosystem", "bandwidth", "move the needle",
    "circle back", "deep dive", "boil the ocean", "low-hanging fruit",
]

PASSIVE_PHRASES = [
    "was responsible for managing",
    "was involved in the development of",
    "duties included the maintenance of",
    "helped in the creation of",
    "assisted in the implementation of",
    "tasks were completed related to",
]

JOB_ROLES = [
    "Software Engineer", "Senior Software Engineer", "Full Stack Developer",
    "Frontend Engineer", "Backend Engineer", "DevOps Engineer",
    "Data Scientist", "ML Engineer", "Cloud Architect",
    "Product Manager", "Engineering Manager", "Tech Lead",
    "SRE Engineer", "Security Engineer", "Mobile Developer",
]

# ─── Generator Functions ───────────────────────────────────────────────────────

def rand(lst, n=1):
    return random.sample(lst, min(n, len(lst)))

def rand1(lst):
    return random.choice(lst)


def generate_quantified_bullet(verb, quality='high'):
    """Generate a bullet point with or without quantification."""
    templates_high = [
        f"{verb} system that reduced latency by {random.randint(20,80)}%",
        f"{verb} platform serving {random.randint(100,999)}K+ monthly active users",
        f"{verb} team of {random.randint(3,15)} engineers across {random.randint(2,5)} time zones",
        f"{verb} infrastructure saving ${random.randint(50,500)}K annually",
        f"{verb} pipeline that increased deployment frequency by {random.randint(2,10)}x",
        f"{verb} API handling {random.randint(1,50)}M+ requests per day",
        f"{verb} feature adopted by {random.randint(60,99)}% of users within 30 days",
        f"{verb} test coverage from {random.randint(20,50)}% to {random.randint(80,99)}%",
        f"{verb} load time by {random.randint(30,70)}% through optimization",
        f"{verb} {random.randint(3,20)} microservices using {rand1(TECH_SKILLS)}",
    ]
    templates_low = [
        f"{verb} various software components",
        f"{verb} tasks assigned by manager",
        f"{verb} code for internal tools",
        f"{verb} backend systems",
        f"{verb} some features for the product",
        f"{verb} documentation and code reviews",
    ]
    templates_weak = [
        f"{rand1(WEAK_VERBS)} the {rand1(['backend', 'frontend', 'database', 'API'])} team",
        f"{rand1(WEAK_VERBS)} various {rand1(['projects', 'tasks', 'features'])}",
        f"{rand1(PASSIVE_PHRASES)} the {rand1(['project', 'product', 'system'])}",
    ]
    if quality == 'high':
        return rand1(templates_high)
    elif quality == 'medium':
        return rand1(templates_low)
    else:
        return rand1(templates_weak)


def generate_resume(quality_level):
    """
    Generate a synthetic resume text.
    quality_level: 'excellent'|'good'|'average'|'poor'|'very_poor'
    Returns dict with resume_text and ground_truth scores.
    """
    q = quality_level

    # Quality parameters
    params = {
        'excellent': dict(
            num_skills=random.randint(10, 16),
            num_exp=random.randint(2, 4),
            bullets_per_exp=random.randint(4, 6),
            bullet_quality='high',
            has_summary=True,
            has_github=True,
            has_linkedin=True,
            has_certs=random.randint(2, 4),
            has_projects=random.randint(2, 3),
            use_buzzwords=0,
            summary_length='long',
            gpa=round(random.uniform(3.5, 4.0), 1),
        ),
        'good': dict(
            num_skills=random.randint(7, 12),
            num_exp=random.randint(2, 3),
            bullets_per_exp=random.randint(3, 4),
            bullet_quality='high' if random.random() > 0.3 else 'medium',
            has_summary=True,
            has_github=random.random() > 0.4,
            has_linkedin=True,
            has_certs=random.randint(1, 2),
            has_projects=random.randint(1, 2),
            use_buzzwords=random.randint(0, 1),
            summary_length='medium',
            gpa=round(random.uniform(3.0, 3.8), 1),
        ),
        'average': dict(
            num_skills=random.randint(4, 8),
            num_exp=random.randint(1, 3),
            bullets_per_exp=random.randint(2, 3),
            bullet_quality='medium',
            has_summary=random.random() > 0.4,
            has_github=random.random() > 0.6,
            has_linkedin=random.random() > 0.5,
            has_certs=random.randint(0, 1),
            has_projects=random.randint(0, 1),
            use_buzzwords=random.randint(1, 3),
            summary_length='short',
            gpa=round(random.uniform(2.7, 3.4), 1),
        ),
        'poor': dict(
            num_skills=random.randint(2, 5),
            num_exp=random.randint(1, 2),
            bullets_per_exp=random.randint(1, 2),
            bullet_quality='low',
            has_summary=random.random() > 0.6,
            has_github=False,
            has_linkedin=random.random() > 0.7,
            has_certs=0,
            has_projects=0,
            use_buzzwords=random.randint(3, 6),
            summary_length='vague',
            gpa=round(random.uniform(2.0, 3.0), 1),
        ),
        'very_poor': dict(
            num_skills=random.randint(1, 3),
            num_exp=random.randint(0, 1),
            bullets_per_exp=random.randint(0, 1),
            bullet_quality='weak',
            has_summary=False,
            has_github=False,
            has_linkedin=False,
            has_certs=0,
            has_projects=0,
            use_buzzwords=random.randint(4, 8),
            summary_length='none',
            gpa=round(random.uniform(1.5, 2.5), 1),
        ),
    }[q]

    name = f"{rand1(['Alex','Jordan','Sam','Taylor','Morgan','Riley','Casey','Drew','Avery','Quinn'])} {rand1(['Johnson','Smith','Williams','Brown','Davis','Wilson','Martinez','Garcia','Lee','Chen'])}"
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@{rand1(['gmail','outlook','yahoo','email'])}.com"
    phone = f"+1 ({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"
    location = rand1(["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Boston, MA", "Chicago, IL", "Remote", "Denver, CO"])

    lines = [name, email, phone, location]
    if params['has_linkedin']:
        lines.append(f"linkedin.com/in/{name.split()[0].lower()}{name.split()[1].lower()}")
    if params['has_github']:
        lines.append(f"github.com/{name.split()[0].lower()}{random.randint(10,99)}")
    lines.append("")

    # Summary
    summaries = {
        'long': f"Senior software engineer with {random.randint(5,10)}+ years of experience building scalable distributed systems and leading cross-functional teams. Proven track record delivering high-impact solutions at scale. Expert in {rand1(TECH_SKILLS)}, {rand1(TECH_SKILLS)}, and cloud architecture. Passionate about developer experience and engineering excellence.",
        'medium': f"Software engineer with {random.randint(2,5)} years experience in {rand1(TECH_SKILLS)} and {rand1(TECH_SKILLS)}. Strong background in full-stack development and agile environments.",
        'short': f"Developer with experience in {rand1(TECH_SKILLS)}. Looking for new opportunities.",
        'vague': f"Hardworking and dedicated professional seeking a challenging position to utilize my skills and grow professionally in a dynamic environment.",
        'none': '',
    }
    summary_text = summaries[params['summary_length']]
    if summary_text:
        if params['use_buzzwords'] > 2:
            buzzword_inject = ' '.join(rand(BUZZWORDS, min(params['use_buzzwords'], 3)))
            summary_text += f" Known for {buzzword_inject}."
        lines += ["SUMMARY", summary_text, ""]

    # Experience
    lines.append("EXPERIENCE")
    skills_chosen = rand(TECH_SKILLS, params['num_skills'])
    exp_skills_used = []

    for e in range(params['num_exp']):
        title = rand1(JOB_ROLES)
        company = rand1(COMPANIES)
        start_year = random.randint(2016, 2022)
        end = "Present" if e == 0 else str(start_year + random.randint(1, 3))
        lines.append(f"{title} | {company} | {start_year}–{end}")
        for _ in range(params['bullets_per_exp']):
            verb = rand1(STRONG_VERBS)
            bullet = generate_quantified_bullet(verb, params['bullet_quality'])
            skill = rand1(TECH_SKILLS)
            exp_skills_used.append(skill)
            if params['bullet_quality'] in ('high', 'medium'):
                bullet += f" using {skill}"
            lines.append(f"• {bullet}")
        lines.append("")

    # Education
    lines.append("EDUCATION")
    degree = rand1(DEGREES)
    school = rand1(UNIVERSITIES)
    grad_year = random.randint(2015, 2023)
    gpa_str = f" | GPA: {params['gpa']}" if params['gpa'] >= 3.0 else ""
    lines.append(f"{degree} | {school} | {grad_year}{gpa_str}")
    lines.append("")

    # Skills
    all_skills = list(set(skills_chosen + exp_skills_used[:3]))
    random.shuffle(all_skills)
    lines.append("SKILLS")
    lines.append(f"Technical: {', '.join(all_skills[:params['num_skills']])}")
    if params['num_skills'] >= 6:
        soft = rand(SOFT_SKILLS, random.randint(2, 4))
        lines.append(f"Soft Skills: {', '.join(soft)}")
    lines.append("")

    # Projects
    if params['has_projects']:
        lines.append("PROJECTS")
        for _ in range(params['has_projects']):
            proj_skill = rand1(TECH_SKILLS)
            proj_skill2 = rand1(TECH_SKILLS)
            stars = random.randint(50, 5000)
            lines.append(f"{rand1(['OpenSource Tool','AI Dashboard','Dev Platform','Analytics Engine','CLI Tool'])} ({proj_skill}, {proj_skill2})")
            if params['bullet_quality'] == 'high':
                lines.append(f"  Built tool with {stars}+ GitHub stars serving {random.randint(10,200)} contributors")
            else:
                lines.append(f"  Personal project using {proj_skill}")
        lines.append("")

    # Certifications
    if params['has_certs']:
        lines.append("CERTIFICATIONS")
        for cert in rand(CERTS, params['has_certs']):
            lines.append(f"• {cert}")
        lines.append("")

    resume_text = "\n".join(lines)

    # ── Compute ground-truth scores ─────────────────────────────────────────
    scores = compute_scores(resume_text, params, quality_level)
    return {"text": resume_text, "scores": scores, "quality": quality_level}


def compute_scores(text, params, quality):
    """Compute realistic ground-truth scores based on resume features."""
    text_lower = text.lower()

    # ATS Score
    base_ats = {'excellent': 88, 'good': 72, 'average': 55, 'poor': 38, 'very_poor': 22}[quality]
    ats_noise = random.randint(-6, 6)
    ats_score = max(10, min(98, base_ats + ats_noise))

    # Formatting score — based on sections present
    sections = ['experience', 'education', 'skills']
    optional = ['summary', 'projects', 'certifications']
    fmt_base = sum(12 for s in sections if s in text_lower)
    fmt_base += sum(6 for s in optional if s in text_lower)
    fmt_base += 10 if params.get('has_linkedin') else 0
    fmt_base += 8 if params.get('has_github') else 0
    formatting_score = max(15, min(98, fmt_base + random.randint(-5, 5)))

    # Content score — bullets, quantification
    bullet_count = text.count('•')
    numbers_count = sum(1 for c in text if c.isdigit())
    strong_verb_count = sum(1 for v in STRONG_VERBS if v in text)
    weak_verb_count = sum(1 for v in WEAK_VERBS if v in text_lower)
    content_base = min(50, bullet_count * 5) + min(20, numbers_count // 3) + min(20, strong_verb_count * 3) - weak_verb_count * 5
    content_score = max(10, min(98, content_base + random.randint(-5, 8)))

    # Skills score
    skill_count = sum(1 for s in TECH_SKILLS if s.lower() in text_lower)
    skills_score = max(10, min(98, min(60, skill_count * 5) + random.randint(-5, 10)))

    # Keywords score
    tech_density = skill_count / max(len(text.split()), 1) * 100
    keywords_score = max(10, min(98, int(tech_density * 8) + random.randint(-5, 15)))

    # Readability score
    sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if len(s.strip()) > 10]
    avg_len = np.mean([len(s.split()) for s in sentences]) if sentences else 20
    buzzword_penalty = params.get('use_buzzwords', 0) * 8
    passive_penalty = sum(1 for p in PASSIVE_PHRASES if p in text_lower) * 10
    readability_base = max(10, 90 - max(0, avg_len - 20) * 2 - buzzword_penalty - passive_penalty)
    readability_score = max(10, min(98, readability_base + random.randint(-5, 5)))

    return {
        "ats_score": ats_score,
        "formatting_score": formatting_score,
        "content_score": content_score,
        "skills_score": skills_score,
        "keywords_score": keywords_score,
        "readability_score": readability_score,
        "has_quantified_achievements": numbers_count > 3,
        "strong_verb_count": strong_verb_count,
        "weak_verb_count": weak_verb_count,
        "skill_count": skill_count,
        "bullet_count": bullet_count,
        "buzzword_count": params.get('use_buzzwords', 0),
    }


def generate_dataset(n=5000):
    """Generate n labeled resume samples with balanced quality distribution."""
    random.seed(42)
    np.random.seed(42)

    # Quality distribution: more average/good to reflect real world
    distribution = {
        'excellent': int(n * 0.10),   # 500  — top 10%
        'good':      int(n * 0.25),   # 1250 — good candidates
        'average':   int(n * 0.35),   # 1750 — most candidates
        'poor':      int(n * 0.20),   # 1000 — weak resumes
        'very_poor': int(n * 0.10),   # 500  — very weak
    }

    dataset = []
    for quality, count in distribution.items():
        print(f"  Generating {count} '{quality}' resumes...")
        for _ in range(count):
            sample = generate_resume(quality)
            dataset.append(sample)

    # Shuffle
    random.shuffle(dataset)
    print(f"\n✓ Generated {len(dataset)} total samples")
    return dataset


def generate_jd_resume_pairs(n=1000):
    """Generate job description + resume pairs for ATS matching training."""
    pairs = []
    jd_templates = [
        "We are looking for a {role} with {n}+ years of experience in {skill1}, {skill2}, and {skill3}. Must have experience with {skill4} and {skill5}. Strong understanding of {skill6} required.",
        "Join our team as a {role}. Required: {skill1}, {skill2}, {skill3}. Nice to have: {skill4}, {skill5}. Experience with {skill6} is a plus.",
        "{role} needed. Must know {skill1} and {skill2}. Experience in {skill3}, {skill4}, {skill5} strongly preferred. {skill6} knowledge required.",
    ]
    for _ in range(n):
        role = rand1(JOB_ROLES)
        skills = rand(TECH_SKILLS, 6)
        jd = rand1(jd_templates).format(
            role=role, n=random.randint(2, 7),
            skill1=skills[0], skill2=skills[1], skill3=skills[2],
            skill4=skills[3], skill5=skills[4], skill6=skills[5]
        )
        quality = rand1(['excellent', 'good', 'average', 'poor'])
        resume = generate_resume(quality)

        # Compute match score based on skill overlap
        jd_lower = jd.lower()
        resume_lower = resume['text'].lower()
        jd_skills = [s for s in skills if s.lower() in jd_lower]
        matched = [s for s in jd_skills if s.lower() in resume_lower]
        match_pct = int((len(matched) / max(len(jd_skills), 1)) * 100)

        pairs.append({
            "resume_text": resume['text'],
            "job_description": jd,
            "match_score": match_pct,
            "matched_skills": matched,
            "missing_skills": [s for s in jd_skills if s not in matched],
        })
    return pairs


if __name__ == "__main__":
    print("Testing data generator...")
    sample = generate_resume('excellent')
    print(sample['text'][:500])
    print("\nScores:", sample['scores'])
