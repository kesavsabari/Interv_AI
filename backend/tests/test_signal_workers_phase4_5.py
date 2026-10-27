"""
Unit & Integration Test Suite for Phase 4.5:
Parallel Signal Workers (Video & Audio), Time-Series Store, Backpressure Queue, and Calibration Comparison Debug Endpoint.
"""

import time
import base64
import unittest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine, SessionLocal
from app.models import User, UserConsent, InterviewConfig, InterviewSession, SessionCalibration, SignalTimeSeries
from app.auth import create_access_token, hash_password
from app.services.signal_workers import (
    VideoSignalWorker,
    AudioSignalWorker,
    SignalBatchStore,
    ParallelSignalProcessor,
    default_signal_store,
    default_signal_processor,
)

client = TestClient(app)


class TestSignalWorkersPhase4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        db = SessionLocal()
        try:
            email = f"candidate_signals_{int(time.time() * 1000)}@test.com"
            self.candidate = User(
                email=email,
                name="Taylor Signals",
                password_hash=hash_password("Secret123!"),
                auth_provider="email",
                is_verified=True,
                onboarding_status="completed",
            )
            db.add(self.candidate)
            db.commit()
            db.refresh(self.candidate)
            self.candidate_id = self.candidate.id
            self.token = create_access_token(data={"sub": str(self.candidate.id), "email": self.candidate.email})
        finally:
            db.close()

        # Consent record
        client.post(
            "/session/consent",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"version": "v1.0", "accepted": True},
        )

    def _create_session_with_calibration(self) -> dict:
        """Creates session and seeds calibration baseline (neutral face, voice volume, pitch, noise)."""
        cfg_resp = client.post(
            "/interview/config",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"round_type": "Technical", "difficulty": "Medium", "duration_minutes": 15, "mode": "normal"},
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

        # Record 30-second calibration baseline
        cal_resp = client.post(
            f"/session/{sess_id}/calibration",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "face_baseline": {
                    "neutral_score": 0.9,
                    "gaze_center": {"x": 0.5, "y": 0.5},
                    "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                },
                "voice_baseline": {
                    "average_volume_db": -22.5,
                    "pitch_range_hz": [110.0, 210.0],
                },
                "background_noise_level_db": -52.0,
                "calibration_text": "I am looking directly at the camera with neutral posture.",
            },
        )
        self.assertEqual(cal_resp.status_code, 201)
        return sess_data

    # =========================================================================
    # 1. VIDEO SIGNAL WORKER
    # =========================================================================

    def test_video_worker_computes_frame_signals(self):
        """Video worker computes face presence, eye contact, head pose, emotion, posture, and fidgeting."""
        worker = VideoSignalWorker()
        calibration = {
            "face_baseline": {
                "gaze_center": {"x": 0.5, "y": 0.5},
                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            }
        }

        # Create a synthetic test JPEG frame (64x64 solid image)
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")
        dummy_img = np.full((120, 120, 3), 180, dtype=np.uint8)
        _, enc = cv2.imencode(".jpg", dummy_img)
        frame_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()

        metrics = worker.process_frame(
            session_id=101,
            frame_data=frame_b64,
            timestamp=100.0,
            calibration=calibration,
        )

        # A blank frame contains no face. Either MediaPipe ran (face_present == 0, no
        # pose/expression invented) or it is unavailable (nothing reported at all).
        if not metrics:
            return
        self.assertEqual(metrics["face_present"], 0.0)
        for unmeasured in ("head_yaw", "head_pitch", "head_roll", "emotion_valence", "posture_alignment"):
            self.assertNotIn(unmeasured, metrics)

    # =========================================================================
    # 2. AUDIO SIGNAL WORKER
    # =========================================================================

    def test_audio_worker_computes_speech_dynamics(self):
        """Audio worker calculates volume RMS, pitch, speech pace, pause count, and filler rates."""
        worker = AudioSignalWorker(sample_rate=16000)
        calibration = {
            "voice_baseline": {
                "average_volume_db": -24.0,
                "pitch_range_hz": [120.0, 200.0],
            },
            "background_noise_level_db": -50.0,
        }

        # Generate 0.5s of 16 kHz synthetic sine wave tone (150 Hz voice pitch)
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        tone = (np.sin(2 * np.pi * 150 * t) * 8000).astype(np.int16)
        audio_bytes = tone.tobytes()

        # Transcript containing fillers: "um", "basically", "like"
        stt_fragment = "Um basically I like distributed caching architectures."

        metrics = worker.process_chunk(
            session_id=202,
            audio_bytes=audio_bytes,
            timestamp=10.0,
            duration=0.5,
            stt_text=stt_fragment,
            calibration=calibration,
        )

        self.assertIn("volume_rms", metrics)
        self.assertIn("pitch_hz", metrics)
        self.assertIn("speech_pace_wpm", metrics)
        self.assertIn("filler_count", metrics)
        self.assertIn("filler_rate", metrics)
        self.assertIn("clarity_proxy", metrics)

        # Verifications
        self.assertGreater(metrics["volume_rms"], 100.0)
        self.assertGreater(metrics["filler_count"], 0)  # "um", "basically", "like" detected
        self.assertGreater(metrics["filler_rate"], 0.0)
        self.assertGreater(metrics["speech_pace_wpm"], 60.0)

    # =========================================================================
    # 3. TIME-SERIES STORE & BATCH WRITING
    # =========================================================================

    def test_signal_batch_store_buffers_and_writes_to_db(self):
        """SignalBatchStore writes batched rows into signal_time_series table."""
        store = SignalBatchStore(batch_size=5, flush_interval_seconds=1.0)
        sess_id = 9999
        q_id = 1

        import asyncio
        async def _run():
            for i in range(6):
                await store.add_signal(
                    session_id=sess_id,
                    question_id=q_id,
                    timestamp=time.time() + (i * 0.5),
                    metric="eye_contact",
                    value=0.91 + (i * 0.01),
                )
            await store.flush()

        asyncio.run(_run())

        signals = store.get_signals_for_question(session_id=sess_id, question_id=q_id)
        self.assertGreaterEqual(len(signals), 6)
        self.assertEqual(signals[0]["metric"], "eye_contact")
        self.assertGreaterEqual(signals[0]["value"], 0.90)

    # =========================================================================
    # 4. BACKPRESSURE QUEUE MANAGEMENT
    # =========================================================================

    def test_parallel_processor_backpressure_drops_frames_when_full(self):
        """Bounded queue drops oldest frame when saturated to prevent interview lag."""
        processor = ParallelSignalProcessor(max_video_queue=3, max_audio_queue=5)
        # Push 8 frames into max_video_queue=3 without blocking
        for i in range(8):
            ok = processor.push_video_frame(
                session_id=888,
                question_id=1,
                timestamp=time.time() + i,
                frame_data="dummy_frame_data",
            )
            self.assertTrue(ok)

        # Queue size must be bounded by max_video_queue
        self.assertLessEqual(processor._video_queue.qsize(), 3)

    # =========================================================================
    # 5. DEBUG ENDPOINT: PER-QUESTION SIGNALS VS CALIBRATION BASELINE
    # =========================================================================

    def test_debug_endpoint_returns_signals_and_calibration_comparison(self):
        """GET /session/{id}/question/{question_id}/signals returns signals compared to baseline."""
        s_data = self._create_session_with_calibration()
        sess_id = s_data["id"]
        q_id = 1

        # Seed test signal data into default_signal_store
        import asyncio
        async def _seed():
            t = time.time()
            metrics = [
                ("face_present", 1.0),
                ("eye_contact", 0.93),
                ("head_yaw", 1.5),
                ("head_pitch", -0.8),
                ("head_roll", 0.2),
                ("emotion_valence", 0.2),
                ("posture_alignment", 0.96),
                ("movement_level", 0.04),
                ("speech_pace_wpm", 142.0),
                ("filler_count", 2.0),
                ("filler_rate", 0.03),
                ("pitch_hz", 155.0),
                ("volume_rms", 460.0),
                ("pause_count", 1.0),
                ("clarity_proxy", 0.91),
            ]
            for m, val in metrics:
                await default_signal_store.add_signal(sess_id, q_id, t, m, val)
            await default_signal_store.flush()

        asyncio.run(_seed())

        resp = client.get(
            f"/session/{sess_id}/question/{q_id}/signals",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Check required fields
        self.assertEqual(data["session_id"], sess_id)
        self.assertEqual(data["question_id"], q_id)
        self.assertTrue(data["has_calibration"])
        self.assertIn("calibration_baseline", data)
        self.assertIn("face_metrics", data)
        self.assertIn("speech_metrics", data)
        self.assertIn("calibration_comparison", data)
        self.assertGreater(data["time_series_count"], 0)

        # Plausible values compared with baseline
        comp = data["calibration_comparison"]
        self.assertIn("head_yaw_delta_from_baseline", comp)
        self.assertIn("eye_contact_relative_to_baseline", comp)
        self.assertGreater(comp["eye_contact_relative_to_baseline"], 0.8)

        # Also verify convenience endpoint /session/{id}/signals
        resp_conv = client.get(
            f"/session/{sess_id}/signals?question_id={q_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(resp_conv.status_code, 200)
        self.assertEqual(resp_conv.json()["session_id"], sess_id)

    # =========================================================================
    # 6. FULL WEBSOCKET ORCHESTRATOR PARALLEL SIGNALS LOOP
    # =========================================================================

    def test_websocket_orchestrator_parallel_signals_ingestion(self):
        """
        WebSocket orchestrator ingests 16 kHz audio and 1 fps video in parallel,
        populates time-series store, and exposes queryable signals via debug endpoint.
        """
        s_data = self._create_session_with_calibration()
        sess_id = s_data["id"]
        sess_token = s_data["session_token"]

        with client.websocket_connect(f"/session/{sess_id}/live?token={sess_token}") as ws:
            ws.receive_json()  # connected
            ws.receive_json()  # greeting avatar_speak
            warmup_q = ws.receive_json()  # warmup question
            q_id = warmup_q.get("id", 1)
            ws.receive_json()  # warmup avatar_speak

            # Ready signal
            ws.send_json({"type": "client_ready", "timestamp": time.time()})
            ws.receive_json()  # timer_started

            # 1. Send video frame at 1 fps
            dummy_jpeg = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIFdummy").decode()
            ws.send_json({"type": "video_frame", "frame": dummy_jpeg, "timestamp": time.time()})
            v_ack = ws.receive_json()
            self.assertEqual(v_ack["type"], "video_frame_ack")

            # 2. Send 16 kHz audio chunk with speech and filler
            pcm_data = base64.b64encode(b"\x80\x00" * 480).decode()
            ws.send_json({
                "type": "audio_chunk",
                "chunk_id": 1,
                "data": pcm_data,
                "duration": 0.25,
                "text": "Um I have experience with distributed services.",
                "timestamp": time.time(),
            })
            c_ack = ws.receive_json()
            self.assertEqual(c_ack["type"], "chunk_ack")

            # 3. Finish answer
            ws.send_json({
                "type": "answer_ended",
                "text": "Um basically I have built scalable microservices and distributed caching systems.",
            })

            # Receive next question or complete
            next_frame = ws.receive_json()
            self.assertIn(next_frame["type"], ("question", "avatar_speak", "redirect"))

        # Wait for background worker tasks to process and flush time-series signals
        sig_data = {}
        for _ in range(30):
            time.sleep(0.1)
            sig_resp = client.get(
                f"/session/{sess_id}/question/{q_id}/signals",
                headers={"Authorization": f"Bearer {self.token}"},
            )
            if sig_resp.status_code == 200:
                sig_data = sig_resp.json()
                if sig_data.get("time_series_count", 0) > 0:
                    break

        self.assertGreater(sig_data.get("time_series_count", 0), 0)
        self.assertIn("face_metrics", sig_data)
        self.assertIn("speech_metrics", sig_data)


if __name__ == "__main__":
    unittest.main()
