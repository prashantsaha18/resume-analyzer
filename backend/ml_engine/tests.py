import numpy as np
from django.test import TestCase
from ml_engine.inference import extract_features, _load_meta, compute_ats_match, analyze_skill_gap
from ml_engine.inference_lite import predict_scores_lite, full_analysis_lite
from ml_engine import router


class MLEngineTestCase(TestCase):
    def setUp(self):
        self.sample_resume = """
        Sam Smith
        sam.smith@email.com
        +1 (555) 123-4567
        San Francisco, CA
        github.com/samsmith
        linkedin.com/in/samsmith

        SUMMARY
        Highly motivated Software Engineer with 5+ years of experience building scalable systems using Python and React. 
        Proven track record of engineering excellence and mentoring peers.

        EXPERIENCE
        Senior Software Engineer | Google | 2022–Present
        • Led team of 5 engineers to build a new data pipeline serving 500K+ monthly active users, using Docker.
        • Optimized backend system performance, reducing latency by 40% using PostgreSQL.
        • Spearheaded migration of microservices to AWS, saving $120K annually.

        Software Engineer | Stripe | 2020–2022
        • Developed responsive user interfaces in React and TypeScript.
        • Automated CI/CD pipelines reducing deployment time by 50% using Git.
        • Worked on various integration tasks with payment gateways.

        EDUCATION
        B.S. Computer Science | Stanford University | 2020 | GPA: 3.8

        SKILLS
        Technical: Python, React, Docker, PostgreSQL, AWS, TypeScript, Git, Linux
        Soft Skills: Leadership, Communication, Mentoring, Agile

        PROJECTS
        OpenSource Tool (Python, Docker)
          Built tool with 1200+ GitHub stars serving 50 contributors.
        """
        
        self.sample_jd = """
        Looking for a Senior Software Engineer with experience in Python, React, Docker, and AWS. 
        Must have excellent communication skills. Experience with system design is a plus.
        """

    def test_load_meta_presets(self):
        """Test vocabulary metadata loader fallback presets."""
        meta = _load_meta()
        self.assertIsNotNone(meta)
        self.assertIn("tech_skills", meta)
        self.assertIn("top_companies", meta)
        self.assertIn("sections", meta)
        self.assertIn("_tech_skills_lower", meta)
        self.assertIn("_top_companies_lower", meta)
        
        # Test company list contains standard presets
        self.assertIn("google", meta["_top_companies_lower"])
        self.assertIn("stripe", meta["_top_companies_lower"])
        self.assertIn("salesforce", meta["_top_companies_lower"])

    def test_extract_features_shape(self):
        """Test that extract_features returns a numpy array with exactly 34 dimensions."""
        features = extract_features(self.sample_resume)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (34,))
        self.assertEqual(features.dtype, np.float32)

    def test_extract_features_correctness(self):
        """Test accuracy of individual feature extractions."""
        features = extract_features(self.sample_resume)
        
        # Feature 5: bullet points count (should find 6 bullet points '•')
        self.assertEqual(features[4], 6.0)
        
        # Feature 6: percentages count (should find 2 '%')
        self.assertEqual(features[5], 2.0)
        
        # Feature 22: contains linkedin (1)
        self.assertEqual(features[21], 1.0)
        
        # Feature 23: contains github (1)
        self.assertEqual(features[22], 1.0)
        
        # Feature 24: contains email (1)
        self.assertEqual(features[23], 1.0)
        
        # Feature 34: company prestige signal (Google, Stripe, and LinkedIn are mentioned, count = 3)
        self.assertEqual(features[33], 3.0)

    def test_ats_match(self):
        """Test compute_ats_match calculates realistic overlap scores."""
        match_data = compute_ats_match(self.sample_resume, self.sample_jd)
        self.assertIn("match_score", match_data)
        self.assertIn("matched_keywords", match_data)
        self.assertIn("missing_critical_keywords", match_data)
        self.assertTrue(0 <= match_data["match_score"] <= 100)
        
        # Should correctly match 'Python' and 'React'
        matched = [k.lower() for k in match_data["matched_keywords"]]
        self.assertIn("python", matched)
        self.assertIn("react", matched)

    def test_skill_gap_analysis(self):
        """Test skill gap analyzer yields appropriate readiness metrics."""
        gap_data = analyze_skill_gap(self.sample_resume, self.sample_jd)
        self.assertIn("overall_readiness", gap_data)
        self.assertIn("matching_skills", gap_data)
        self.assertIn("critical_missing_skills", gap_data)
        self.assertIn("learning_path", gap_data)
        self.assertTrue(0 <= gap_data["overall_readiness"] <= 100)

    def test_predict_scores_lite(self):
        """Test rule-based scoring fallback correctness."""
        scores = predict_scores_lite(self.sample_resume)
        self.assertEqual(len(scores), 6)
        self.assertIn("ats_score", scores)
        self.assertIn("formatting_score", scores)
        self.assertIn("content_score", scores)
        self.assertIn("skills_score", scores)
        self.assertIn("keywords_score", scores)
        self.assertIn("readability_score", scores)
        
        for k, v in scores.items():
            self.assertTrue(10 <= v <= 98, f"{k} score {v} out of bounds")

    def test_full_analysis_lite(self):
        """Test full analysis lite output structure."""
        res = full_analysis_lite(self.sample_resume, self.sample_jd)
        self.assertIn("ats_score", res)
        self.assertIn("found_keywords", res)
        self.assertIn("missing_keywords", res)
        self.assertIn("suggestions", res)
        self.assertIn("action_verbs", res)
        self.assertIn("strong", res["action_verbs"])
        self.assertIn("weak", res["action_verbs"])
        self.assertIn("overall_feedback", res)
        
        # Assert action verbs contain the strong verbs mentioned
        strong = [v.lower() for v in res["action_verbs"]["strong"]]
        self.assertIn("led", strong)
        self.assertIn("optimized", strong)
        self.assertIn("spearheaded", strong)

    def test_router_backends(self):
        """Test backend selector and settings lookup caches."""
        backend = router.get_backend()
        self.assertIn(backend, ["gemini", "ml", "lite"])
        
        # Test cached value is set
        self.assertIsNotNone(router._cached_use_gemini)
        self.assertIsNotNone(router._cached_ml_available)
