"""Score every press release with FinBERT (ProsusAI/finbert) and tag its event type. Output: data/events.csv."""
import os, sys, re, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, ROOT, UNIVERSE
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_TOKENS, BATCH = 256, 32
KINDS = [
    ("fda", re.compile(r"\bFDA\b|approv|PDUFA|complete response letter|\bCRL\b|breakthrough therapy|priority review|\bNDA\b|\bBLA\b|\bsNDA\b|\bsBLA\b|marketing authori[sz]ation|\bEMA\b|\bCHMP\b", re.I)),
    ("trial", re.compile(r"phase\s?[1-3I]|top-?line|pivotal|endpoint|clinical (trial|study|data)|study results|interim (data|analysis)|efficacy|safety data|presents? (new )?data|dose", re.I)),
    ("earnings", re.compile(r"financial results|quarter(ly)? (results|report)|first quarter|second quarter|third quarter|fourth quarter|full[- ]year|annual results|\bQ[1-4]\b", re.I)),
    ("deal", re.compile(r"acqui|merger|collaborat|licens|partnership|agreement|offering|financing|pricing of", re.I)),
]
def kind(title):
    for k, rx in KINDS:
        if rx.search(title or ""): return k
    return "other"

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
torch.set_num_threads(max(1, os.cpu_count() - 1))
labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
ip, ineg = labels.index("positive"), labels.index("negative")

rows = []
for t in UNIVERSE:
    p = RAW / "press_releases" / f"{t}.json"
    if not p.exists(): continue
    for r in json.loads(p.read_text()):
        text = (r.get("title") or "") + ". " + (r.get("text") or "")[:2000]
        rows.append({"ticker": t, "date": r["date"][:10], "time": r["date"][11:16], "title": (r.get("title") or "").strip()[:200], "kind": kind(r.get("title")), "_text": text})
print("releases to score", len(rows), flush=True)

with torch.no_grad():
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        enc = tok([b["_text"] for b in batch], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        prob = torch.softmax(model(**enc).logits, dim=-1)
        for b, pr in zip(batch, prob):
            b["p_pos"], b["p_neg"] = round(float(pr[ip]), 4), round(float(pr[ineg]), 4)
            b["score"] = round(float(pr[ip] - pr[ineg]), 4)
        if (i // BATCH) % 50 == 0: print(i, "/", len(rows), flush=True)

with open(ROOT / "data" / "events.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker", "date", "time", "kind", "p_pos", "p_neg", "score", "title"])
    w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})
print("wrote", len(rows))
