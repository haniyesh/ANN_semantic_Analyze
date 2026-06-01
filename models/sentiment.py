import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import SENTIMENT_MODEL_NAME

_tokenizer = None
_model = None
_device = None


def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    print("[SENTIMENT] Loading FinBERT...")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    _model     = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME).to(_device)
    _model.eval()
    print(f"[SENTIMENT] Loaded on {_device} ✅")


def get_sentiment(title: str) -> str:
    """Returns 'positive', 'negative', or 'neutral'."""
    _load()
    try:
        inputs = _tokenizer(
            title[:512], return_tensors="pt",
            truncation=True, max_length=128,
            padding=True
        ).to(_device)
        with torch.no_grad():
            probs = torch.softmax(_model(**inputs).logits, dim=1)[0]
        # FinBERT order: positive=0, negative=1, neutral=2
        labels = ["positive", "negative", "neutral"]
        return labels[probs.argmax().item()]
    except Exception as e:
        print(f"[SENTIMENT ERROR] {e}")
        return "neutral"


def get_sentiment_full(title: str) -> dict:
    """Returns all scores — useful for debugging."""
    _load()
    try:
        inputs = _tokenizer(
            title[:512], return_tensors="pt",
            truncation=True, max_length=128,
            padding=True
        ).to(_device)
        with torch.no_grad():
            probs = torch.softmax(_model(**inputs).logits, dim=1)[0]
        pos, neg, neu = probs[0].item(), probs[1].item(), probs[2].item()
        confidence = max(pos, neg, neu)
        if pos > neg and pos > neu:
            sentiment = "positive"
        elif neg > pos and neg > neu:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return {
            "sentiment":    sentiment,
            "confidence":   round(confidence, 4),
            "prob_positive": round(pos, 4),
            "prob_negative": round(neg, 4),
            "prob_neutral":  round(neu, 4),
        }
    except Exception as e:
        print(f"[SENTIMENT ERROR] {e}")
        return {"sentiment": "neutral", "confidence": 0.5,
                "prob_positive": 0.0, "prob_negative": 0.0, "prob_neutral": 1.0}