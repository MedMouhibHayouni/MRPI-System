import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1",
    reason="Live LLM smoke test disabled unless RUN_LIVE_LLM_TESTS=1.",
)
def test_llm_connection_smoke():
    load_dotenv(override=True)

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=50,
    )

    assert response.choices[0].message.content
