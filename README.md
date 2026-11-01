# Hotel Review Sentiment Analysis

Classifies TripAdvisor hotel reviews as negative, neutral or positive, comparing a text CNN written in PyTorch with a TF-IDF baseline. There's also a small desktop app in `Base/` built around the original Keras model.

## Results

Mean over 3 random 80/20 splits of 20,491 reviews (1 to 2 stars is negative, 3 is neutral, 4 to 5 is positive):

| Model | Accuracy | Macro F1 |
| :--- | ---: | ---: |
| **TF-IDF with bigrams + logistic regression** | **84.5%** | **0.709** |
| Text CNN (filter widths 3, 4, 5) | 83.2% | 0.684 |

**The simple baseline wins.** On this much data, the CNN doesn't earn its extra training cost, so the baseline is the one I'd deploy.

Neutral is the hard class: F1 is about 0.43 for the baseline against 0.93 for positive, on the first split. Three star reviews mix praise and complaints, and they make up only 11% of the data, which is why macro F1 sits so far below accuracy.

## Run it

```
pip install -r requirements.txt
python train.py
```

Results land in `results/metrics.json`.

## Repo layout

| Path | What it is |
| :--- | :--- |
| [`train.py`](train.py) | Trains and compares both models |
| `Data/` | [TripAdvisor Hotel Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) |
| [`Base/original_keras_model.py`](Base/original_keras_model.py) | The first Keras version of the model |
| [`Base/Working.py`](Base/Working.py) | Desktop app that loads the saved Keras model, and needs `keras`, `textblob`, `customtkinter` and `CTkMessagebox` |
