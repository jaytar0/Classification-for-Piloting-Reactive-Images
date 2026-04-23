# Reactive Image Classification: Twitch Emotion Detection

A multi-class emotion classification pipeline built on Twitch IRC chat data, using emotes as emotion proxies to classify messages across Paul Ekman's emotion taxonomy without manual annotation.

## Overview

The initial aim of this project was to be able to accurately train an emotional classification model based on internet lingo and short messages (twitch chat) with an associated label, emotes. This would be used in an effort to hook up to a live speech to text system to help streamers that use animated or stand-still avatars to toggle different variants of their models when expressing certain emotion, instead of manually toggling it. This project builds a full pipeline from raw IRC data collection to model benchmarking, using a hybrid auto-labeling strategy to avoid manual annotation at scale.

DistilBERT-LoRA outperformed all 6 baseline models, achieving the highest weighted F1 0.388 and macro F1 0.360 despite being trained on only ~7,300 instances, demonstrating the robustness of low-rank adaptation on limited data.

## Pipeline

```
IRC Scraping → Cleaning I → Cleaning II → Auto-Labeling → Modeling → Evaluation
```

**Data Scraping:** Custom Twitch IRC scraper using raw socket connections, collecting messages from 10-15 channels simultaneously. Filters bots, emote-only messages, and spam. Captures native Twitch emotes + 3rd party emotes via 7TV, BTTV, and FrankerFaceZ APIs.

**Cleaning I:** Hybrid auto-labeling pipeline combining:
   - BERT cosine similarity (all-MiniLM-L6-v2) against Ekman emotion vectors
   - Zero-shot classification via BART-large-MNLI on emote tokens
   - Summed weight voting to assign final emotion labels
   - Override dictionary for high-confidence known emotes (e.g. `Sadge` -> sadness, `KEKW` -> happiness)

**Cleaning II:** Outlier removal, copy-pasta filtering, class distribution balancing, final text normalization

**Modeling:** 7 models benchmarked across 4 emotion classes (happiness, neutral, sadness, surprise)

**Evaluation:** F1, precision, recall, confusion matrices output to LaTeX


## Results

| Model | Weighted F1 | Macro F1 |
|---|---|---|
| **DistilBERT-LoRA** | **0.388** | **0.360** |
| TF-IDF: Logistic Regression | 0.373 | 0.359 |
| FastText | 0.373 | 0.336 |
| TF-IDF: SVM | 0.362 | 0.312 |
| TF-IDF: Random Forest | 0.356 | 0.321 |
| Naive Bayes + BoW | 0.342 | 0.280 |
| TF-IDF: XGBoost | 0.332 | 0.278 |

DistilBERT-LoRA was fine-tuned with `r=16`, `lora_alpha=64`, targeting `q_lin`, `v_lin`, `k_lin` attention layers with frozen base weights. Results show expected overfitting given smaller training samples, performance would scale significantly with more data.

---

### Tech Used

**Scraping:** Python, socket, Twitch IRC
**NLP/Labeling:** HuggingFace Transformers, sentence-transformers, BART-large-MNLI
**Modeling:** DistilBERT-LoRA (PEFT), FastText, Scikit-learn, XGBoost
**Data:** Pandas, NumPy, regex, emoji
**Evaluation:** Scikit-learn metrics, Matplotlib, Seaborn, LaTeX output



### Dataset

Labeled instances across 4 emotion classes after cleaning and balancing. Raw data collected from Twitch IRC across multiple channels. Data not included in repo due to size, scraper and pipeline scripts are fully reproducible. Test set holdout is below.

| Emotion | Count |
|---|---|
| Happiness | 501 |
| Surprise | 501 |
| Neutral | 296 |
| Sadness | 171 |



## Limitations & Future Work

- LoRA models perform best with significantly more data (50K+ recommended). Current results are promising given scope constraints.
- Emote-based labeling captures dominant emotions but misses emotional duality and subtlety
- Future direction: incorporate streamer speech-to-text as contextual signal for more accurate labeling
- Expanding back to all 7 Ekman categories with sufficient class representation per emotion
