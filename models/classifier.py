import json
from groq import Groq
from config import GROQ_API_KEYS, GROQ_CLASSIFICATION_MODEL

# Build list of Groq clients from multiple API keys
_clients = [Groq(api_key=key) for key in GROQ_API_KEYS if key]


def _call_groq(prompt: str, max_tokens: int = 200) -> str | None:
    """Try each API key in order — rotate on rate limit."""
    for client in _clients:
        try:
            response = client.chat.completions.create(
                model=GROQ_CLASSIFICATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            if raw:
                return raw
        except Exception as e:
            if "429" in str(e):
                print("[CLASSIFIER] Rate limited, trying next key...")
                continue
            raise e
    return None


def classify_news(text: str) -> dict:
    """Classify a single news item."""
    return classify_news_batch([text])[0]


def classify_news_batch(texts: list) -> list:
    """
    Classify multiple news items in one API call.
    Returns list of dicts: {type, impact_horizon, confidence}
    """
    fallback = {"type": "macro_geopolitical", "impact_horizon": "1h", "confidence": 0.5}

    numbered = "\n\n".join([f"{i}. {text[:200]}" for i, text in enumerate(texts)])
    prompt = f"""Classify each news item into one of:
[micro_crypto, macro_geopolitical, regulation, institutional_flow, onchain, hype]

Return ONLY a JSON array with one object per item in order:
[{{"type": "...", "impact_horizon": "...", "confidence": 0.0}}, ...]

impact_horizon options: 15min, 1h, 4h, 1d
confidence: 0.0 to 1.0

News items:
{numbered}
JSON array:"""

    try:
        raw = _call_groq(prompt)
        if not raw:
            return [fallback] * len(texts)

        start = raw.find("[")
        if start == -1:
            return [fallback] * len(texts)

        depth, end = 0, start
        for i, c in enumerate(raw[start:], start):
            if c == "[":   depth += 1
            elif c == "]": depth -= 1
            if depth == 0:
                end = i + 1
                break

        results = json.loads(raw[start:end])
        while len(results) < len(texts):
            results.append(fallback)
        return results

    except Exception as e:
        print(f"[CLASSIFIER ERROR] {e}")
        return [fallback] * len(texts)