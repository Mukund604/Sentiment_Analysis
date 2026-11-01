"""Trains a text CNN on TripAdvisor reviews and compares it with a TF-IDF baseline.

Usage: python train.py   (reads Data/, writes results/ and models/)
Neutral reviews are the smallest class, so macro F1 sits next to accuracy.
"""
import json
import pathlib
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F

LABELS = ["Negative", "Neutral", "Positive"]
MAX_LEN, VOCAB = 200, 20000
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    frame = pd.read_csv(pathlib.Path(__file__).parent / "Data/tripadvisor_hotel_reviews.csv")
    frame = frame.drop_duplicates("Review")
    # 1 to 2 stars is negative, 3 is neutral, 4 to 5 is positive.
    y = np.select([frame.Rating < 3, frame.Rating == 3], [0, 1], 2)
    return frame.Review.to_numpy(), y


def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


class Encoder:
    def __init__(self, texts):
        counts = Counter(t for text in texts for t in tokenize(text))
        self.index = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB - 2))}

    def __call__(self, texts):
        out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = [self.index.get(t, 1) for t in tokenize(text)][:MAX_LEN]
            out[i, : len(ids)] = ids
        return out


class TextCNN(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 128, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(128, 128, k) for k in (3, 4, 5))
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(128 * 3, classes)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)
        pooled = [F.relu(c(x)).amax(dim=2) for c in self.convs]
        return self.head(self.drop(torch.cat(pooled, 1)))


def predict(model, ids):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(ids[i:i + 256]).to(DEVICE)).argmax(1).cpu().numpy()
                               for i in range(0, len(ids), 256)])


def train_cnn(x_tr, y_tr, seed, epochs=25, patience=4):
    torch.manual_seed(seed)
    x_fit, x_val, y_fit, y_val = train_test_split(x_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=seed)
    model = TextCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weight = torch.tensor(np.sqrt(len(y_fit) / (3 * np.bincount(y_fit))), dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    best, state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_fit))
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            loss = F.cross_entropy(model(torch.from_numpy(x_fit[batch]).to(DEVICE)),
                                   torch.from_numpy(y_fit[batch]).to(DEVICE), weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val = f1_score(y_val, predict(model, x_val), average="macro")
        if val > best:
            best, wait = val, 0
            state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return model


def scores(y, p):
    return dict(accuracy=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"))


def main():
    texts, y = load()
    rows = {"baseline": [], "cnn": []}
    report = None
    for seed in range(3):
        a, b, y_a, y_b = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=seed)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        base = LogisticRegression(max_iter=2000, class_weight="balanced").fit(tfidf.fit_transform(a), y_a)
        base_pred = base.predict(tfidf.transform(b))
        rows["baseline"].append(scores(y_b, base_pred))
        encoder = Encoder(a)
        model = train_cnn(encoder(a), y_a, seed)
        pred = predict(model, encoder(b))
        rows["cnn"].append(scores(y_b, pred))
        print(seed, rows["baseline"][-1], rows["cnn"][-1], flush=True)
        if seed == 0:
            report = dict(cnn=classification_report(y_b, pred, target_names=LABELS, output_dict=True),
                          baseline=classification_report(y_b, base_pred, target_names=LABELS, output_dict=True))
    summary = {m: {k: dict(mean=float(np.mean([r[k] for r in v])), std=float(np.std([r[k] for r in v])))
                   for k in v[0]} for m, v in rows.items()}
    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(dict(summary=summary, seed_0_report=report,
                                                      reviews=len(y), class_counts=np.bincount(y).tolist()), indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
