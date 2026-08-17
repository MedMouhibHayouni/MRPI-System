# locustfile.py — Test de charge combiné MRPI-System
# ... (your existing docstring remains unchanged)
import logging
import random
from locust import HttpUser, task, between, events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- HARDCODED BASE URL (change this if your server runs elsewhere) ----------
BASE_URL = "http://localhost:8000"  # <-- REMOVE the 'clear' here!
# If you want to still allow --host override, you can keep the --host flag,
# but this ensures a fallback. I strongly recommend fixing the --host command instead.

NO_LLM_PATH = "/recommendations/no-llm"
LLM_PATH = "/recommendations"

FAILURE_THRESHOLDS_MS = {
    NO_LLM_PATH: 2000,
    LLM_PATH: 5000,
}
MAX_FAILURE_RATIO = 0.01

STUDENT_IDS = [f"STU-2026-{i:04d}" for i in range(1, 201)]
SUBJECTS = ["math", "physics", "chemistry", "biology"]
WEAK_CONCEPTS = ["algebra", "mechanics", "stoichiometry", "genetics", "geometry"]
ACADEMIC_LEVELS = ["beginner", "intermediate", "advanced"]
LEARNING_STYLES = ["textual", "visual", "auditory", "kinesthetic"]


def build_recommendation_payload() -> dict:
    return {
        "student_id": random.choice(STUDENT_IDS),
        "subject": random.choice(SUBJECTS),
        "weak_concept": random.choice(WEAK_CONCEPTS),
        "academic_level": random.choice(ACADEMIC_LEVELS),
        "learning_style": random.choice(LEARNING_STYLES),
        "past_interactions": [],
    }


def validate_recommendation_response(resp, check_rate_limit: bool = False) -> None:
    if resp.status_code == 404:
        resp.success()
        return
    if check_rate_limit and resp.status_code == 429:
        resp.failure("429 — rate limit LLM atteint malgré le throttling")
        return
    if resp.status_code != 200:
        resp.failure(f"Statut inattendu: {resp.status_code} — {resp.text[:200]}")
        return
    try:
        data = resp.json()
    except ValueError:
        resp.failure("Réponse non-JSON")
        return
    required_fields = [
        "request_id",
        "student_id",
        "recommendations",
        "llm_explanation",
        "metadata",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        resp.failure(f"Champs obligatoires manquants: {missing}")
        return
    if (
        not isinstance(data["recommendations"], list)
        or len(data["recommendations"]) == 0
    ):
        resp.failure("'recommendations' vide ou de mauvais type")
        return
    resp.success()


class NoLLMUser(HttpUser):
    wait_time = between(0.1, 0.5)
    weight = 1

    @task
    def get_recommendations_no_llm(self):
        payload = build_recommendation_payload()
        # Use BASE_URL + path explicitly
        with self.client.post(
            BASE_URL + NO_LLM_PATH, json=payload, name=NO_LLM_PATH, catch_response=True
        ) as resp:
            validate_recommendation_response(resp)


class LLMThrottledUser(HttpUser):
    wait_time = between(20, 30)
    weight = 1

    @task
    def get_recommendations_with_llm(self):
        payload = build_recommendation_payload()
        with self.client.post(
            BASE_URL + LLM_PATH, json=payload, name=LLM_PATH, catch_response=True
        ) as resp:
            validate_recommendation_response(resp, check_rate_limit=True)


class LLMStressUser(HttpUser):
    wait_time = between(1, 3)
    weight = 0  # keep 0 unless you have a paid LLM tier

    @task
    def get_recommendations_with_llm(self):
        payload = build_recommendation_payload()
        with self.client.post(
            BASE_URL + LLM_PATH, json=payload, name=LLM_PATH, catch_response=True
        ) as resp:
            validate_recommendation_response(resp, check_rate_limit=True)


@events.quitting.add_listener
def _check_failure_threshold(environment, **kwargs):
    had_failure = False
    for (name, method), entry in environment.stats.entries.items():
        if entry.num_requests == 0 or name not in FAILURE_THRESHOLDS_MS:
            continue
        fail_ratio = entry.num_failures / entry.num_requests
        p95 = entry.get_response_time_percentile(0.95)
        threshold = FAILURE_THRESHOLDS_MS[name]
        if fail_ratio > MAX_FAILURE_RATIO:
            print(f"❌ [{name}] Taux d'échec trop élevé: {fail_ratio:.2%}")
            had_failure = True
        if p95 and p95 > threshold:
            print(f"❌ [{name}] p95 trop élevé: {p95:.0f} ms (seuil: {threshold} ms)")
            had_failure = True
    if had_failure:
        environment.process_exit_code = 1
