import sys
import os
import time
import json
import unittest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.database import Base, engine, SessionLocal
from app.models import User, TargetProfile, InterviewConfig, UserConsent, InterviewSession, QuestionBankItem
from app.auth import hash_password, create_access_token
from app.services.session_state import default_state_store, SessionState, SessionStateData
from app.services.tts_service import default_tts_service, TTSService, MockTTSProvider, create_minimal_wav_bytes
from app.services.question_selector import select_next_question, QuestionSelectorService

client = TestClient(app)


class TestInterviewerPhase4_2(unittest.TestCase):
    """
    Test suite for Phase 4.2 (Interviewer Side):
    1. Avatar greeting and warm-up (personalized from name & role; first Q always 'Tell me about yourself').
    2. Question selector service (topic weights, difficulty fit, CV item preference, pending follow-up, strict deduplication).
    3. Text to speech (provider abstraction, audio caching for repeated phrases, language selection, audio endpoint).
    4. Playback-end ready handshake (client_ready -> timer_started and recording kickoff).
    5. Full scripted WebSocket session verifying subtitles, speech events, and blueprint progression.
    """

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        db = SessionLocal()
        try:
            self.test_user = User(
                email=f"alex_interviewer_{int(time.time() * 1000)}@test.com",
                name="Alex Kumar",
                password_hash=hash_password("AlexPass123!"),
                auth_provider="email",
                is_verified=True,
                onboarding_status="completed",
            )
            db.add(self.test_user)
            db.commit()
            db.refresh(self.test_user)

            self.target_profile = TargetProfile(
                user_id=self.test_user.id,
                target_role="Senior Fullstack Engineer",
                company_type="Product",
                experience_level="Mid-Senior",
            )
            db.add(self.target_profile)
            db.commit()

            self.token = create_access_token(
                data={"sub": str(self.test_user.id), "email": self.test_user.email}
            )
        finally:
            db.close()

        # Accept consent
        client.post(
            "/session/consent",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"version": "v1.0", "accepted": True},
        )

    # =========================================================================
    # 1. QUESTION SELECTOR TESTS
    # =========================================================================

    def test_question_selector_respects_blueprint_topic_weights(self):
        """Selector prioritizes topics with largest weight deficit."""
        blueprint = {
            "topics": [
                {"name": "System Design", "weight": 0.7},
                {"name": "DSA", "weight": 0.3},
            ],
            "difficulty_curve": ["Medium", "Hard"],
        }
        pool = [
            {"id": 1, "text": "DSA Sliding Window", "topic": "DSA", "difficulty": "Medium", "is_cv_based": False},
            {"id": 2, "text": "Design Rate Limiter", "topic": "System Design", "difficulty": "Medium", "is_cv_based": False},
        ]
        session_context = []

        # System design has weight 0.7 vs 0.3, so it should be picked first
        selected = select_next_question(pool, blueprint, session_context, current_difficulty="Medium")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], 2)
        self.assertEqual(selected["topic"], "System Design")

    def test_question_selector_prefers_cv_based_questions(self):
        """Given equal topic deficit, questions referencing candidate CV receive higher score."""
        blueprint = {
            "topics": [{"name": "System Design", "weight": 1.0}],
            "difficulty_curve": ["Medium"],
        }
        pool = [
            {"id": 10, "text": "Generic DB Sharding", "topic": "System Design", "difficulty": "Medium", "is_cv_based": False},
            {"id": 11, "text": "Kafka pipeline from your Uber clone", "topic": "System Design", "difficulty": "Medium", "is_cv_based": True},
        ]
        session_context = []

        selected = select_next_question(pool, blueprint, session_context, current_difficulty="Medium")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], 11)
        self.assertTrue(selected["is_cv_based"])

    def test_question_selector_strictly_never_repeats_questions(self):
        """Questions already asked in session_context are strictly excluded."""
        blueprint = {
            "topics": [{"name": "DSA", "weight": 1.0}],
            "difficulty_curve": ["Medium"],
        }
        pool = [
            {"id": 201, "text": "Two Sum Problem", "topic": "DSA", "difficulty": "Medium"},
            {"id": 202, "text": "Binary Tree Inversion", "topic": "DSA", "difficulty": "Medium"},
        ]
        # Simulate Q201 already asked
        session_context = [{"question_id": 201, "question": "Two Sum Problem", "topic": "DSA"}]

        selected = select_next_question(pool, blueprint, session_context, current_difficulty="Medium")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], 202)

        # Simulate both asked -> pool exhausted
        session_context.append({"question_id": 202, "question": "Binary Tree Inversion", "topic": "DSA"})
        selected_exhausted = select_next_question(pool, blueprint, session_context, current_difficulty="Medium")
        self.assertIsNone(selected_exhausted)

    def test_question_selector_prioritizes_pending_followup(self):
        """If a pending follow-up is routed, it takes immediate precedence."""
        blueprint = {"topics": [{"name": "DSA", "weight": 1.0}]}
        pool = [{"id": 301, "text": "Sort array", "topic": "DSA"}]
        pending_followup = {
            "id": 999,
            "text": "What is the amortized space complexity of your hash table approach?",
            "topic": "DSA",
            "is_follow_up": True,
        }

        selected = select_next_question(pool, blueprint, [], pending_followup=pending_followup)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], 999)
        self.assertTrue(selected["is_follow_up"])

    # =========================================================================
    # 2. TEXT-TO-SPEECH (TTS) SERVICE TESTS
    # =========================================================================

    def test_tts_service_synthesis_and_caching(self):
        """TTS service caches repeated phrases and returns identical cache_key."""
        tts = TTSService(provider=MockTTSProvider())
        phrase = "Tell me about yourself, your background, and what drew you to this role."

        res1 = tts.synthesize_speech(phrase, language="English")
        self.assertIn("audio_url", res1)
        self.assertIn("cache_key", res1)
        self.assertEqual(res1["language"], "en")

        res2 = tts.synthesize_speech(phrase, language="English")
        self.assertEqual(res1["cache_key"], res2["cache_key"])
        self.assertTrue(res2["is_cached"])

    def test_tts_service_audio_endpoint(self):
        """GET /session/tts/audio/{cache_key} returns streaming audio bytes."""
        phrase = "Welcome to IntervAI live interview session."
        synth = default_tts_service.synthesize_speech(phrase, language="English")
        cache_key = synth["cache_key"]

        resp = client.get(f"/session/tts/audio/{cache_key}")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("audio/", resp.headers["content-type"])
        self.assertGreater(len(resp.content), 0)

    def test_tts_service_language_mapping(self):
        """TTS service correctly maps interview config languages to ISO codes."""
        tts = TTSService(provider=MockTTSProvider())
        spanish = tts.synthesize_speech("Hola", language="Spanish")
        self.assertEqual(spanish["language"], "es")

        hindi = tts.synthesize_speech("Namaste", language="Hindi")
        self.assertEqual(hindi["language"], "hi")

        french = tts.synthesize_speech("Bonjour", language="French")
        self.assertEqual(french["language"], "fr")

    # =========================================================================
    # 3. WEBSOCKET SESSION ORCHESTRATION & HANDSHAKE TESTS
    # =========================================================================

    def test_avatar_greeting_and_warmup_flow(self):
        """
        WebSocket session flow:
        - Personalized greeting emitted via avatar_speak mentioning candidate name & role
        - First warmup question is strictly 'Tell me about yourself'
        - Candidate sends client_ready on playback end -> server starts answer timer
        """
        # 1. Create interview config & session
        cfg_resp = client.post(
            "/interview/config",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "round_type": "Technical",
                "difficulty": "Medium",
                "duration_minutes": 20,
                "mode": "normal",
                "language": "English",
            },
        )
        self.assertEqual(cfg_resp.status_code, 201)
        cfg_id = cfg_resp.json()["id"]

        sess_resp = client.post(
            "/session",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"interview_id": cfg_id},
        )
        self.assertEqual(sess_resp.status_code, 201)
        sess_data = sess_resp.json()
        sess_id = sess_data["id"]

        sess_token = sess_data["session_token"]
        # 2. Connect fake client to WebSocket
        with client.websocket_connect(f"/session/{sess_id}/live?token={sess_token}") as ws:
            # First frame: connected
            f_conn = ws.receive_json()
            self.assertEqual(f_conn["type"], "connected")

            # Second frame: avatar_speak (greeting)
            f_greet = ws.receive_json()
            self.assertEqual(f_greet["type"], "avatar_speak")
            self.assertIn("Alex Kumar", f_greet["text"])
            self.assertIn("Senior Fullstack Engineer", f_greet["text"])
            self.assertTrue(f_greet["audio_url"].startswith("/session/tts/audio/"))

            # Third frame: question (warmup)
            f_warmup_q = ws.receive_json()
            self.assertEqual(f_warmup_q["type"], "question")
            self.assertEqual(f_warmup_q["topic"], "Warmup")
            self.assertIn("Tell me about yourself", f_warmup_q["text"])

            # Fourth frame: avatar_speak (speaking warmup question)
            f_warmup_speak = ws.receive_json()
            self.assertEqual(f_warmup_speak["type"], "avatar_speak")
            self.assertIn("Tell me about yourself", f_warmup_speak["text"])

            # 3. Handshake: client finishes playing audio and sends client_ready
            ws.send_json({"type": "client_ready", "timestamp": time.time()})

            # Server emits timer_started
            f_timer = ws.receive_json()
            self.assertEqual(f_timer["type"], "timer_started")
            self.assertGreater(f_timer["time_limit"], 0)
            self.assertEqual(f_timer["question_id"], 1)

            # 4. Candidate delivers answer
            ws.send_json({
                "type": "answer_ended",
                "text": "I have 5 years experience building distributed React and FastAPI microservices.",
                "timestamp": time.time(),
            })

            # 5. Server evaluates, routes to ASKING, selects next question following blueprint
            # Server emits: question (main technical question)
            # The spoken "Mm-hm." ack runs in parallel with evaluation, so it can arrive anywhere here
            def next_frame():
                f = ws.receive_json()
                while f["type"] == "ack":
                    f = ws.receive_json()
                return f
            f_next_q = next_frame()
            self.assertEqual(f_next_q["type"], "question")
            # Must not be warmup or tell me about yourself
            self.assertNotEqual(f_next_q["topic"], "Warmup")
            self.assertNotIn("Tell me about yourself", f_next_q["text"])

            # Server emits: avatar_speak for next question with audio_url and subtitles
            f_next_speak = next_frame()
            self.assertEqual(f_next_speak["type"], "avatar_speak")
            self.assertEqual(f_next_speak["text"], f_next_q["text"])
            self.assertTrue(f_next_speak["audio_url"].startswith("/session/tts/audio/"))

    def test_websocket_connect_with_jwt_access_token(self):
        """Verifies candidate can connect to WebSocket stream using their JWT access token."""
        cfg_resp = client.post(
            "/interview/config",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"round_type": "Technical", "difficulty": "Medium", "duration_minutes": 15},
        )
        self.assertEqual(cfg_resp.status_code, 201)
        sess_resp = client.post(
            "/session",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"interview_id": cfg_resp.json()["id"]},
        )
        self.assertEqual(sess_resp.status_code, 201)
        sess_id = sess_resp.json()["id"]

        with client.websocket_connect(f"/session/{sess_id}/live?token={self.token}") as ws:
            conn = ws.receive_json()
            self.assertEqual(conn["type"], "connected")
            greet = ws.receive_json()
            self.assertEqual(greet["type"], "avatar_speak")


if __name__ == "__main__":
    unittest.main()

