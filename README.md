# ⚽ Football Ragebait Detector

> *"Messi is finished lmaooo 😂"* — Ragebait. Obviously.
> *"Messi's positioning creates space for teammates"* — Genuine opinion.
>
> But can a machine tell the difference? Turns out — yes. Very confidently.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://ragebait-detector.streamlit.app)
[![Model](https://img.shields.io/badge/Model-BERTweet-blue?style=for-the-badge)](https://huggingface.co/vinai/bertweet-base)
[![Python](https://img.shields.io/badge/Python-3.10-yellow?style=for-the-badge)](https://python.org)

---

## 🤔 Wait, What Even Is Ragebait?

You know that tweet you see at 11pm that says:

> *"Ronaldo hasn't been world class since 2018, change my mind"*

You know it's bait. You know engaging with it is a trap. You do it anyway.

**That's ragebait.** Tweets specifically designed to provoke emotional reactions, trigger arguments, and farm engagement through outrage rather than genuine discussion.

Football Twitter is absolutely drowning in it. This project detects it automatically.

---

## 🆕 Why This Also Helps New Football Fans

If you're new to football, ragebait is genuinely dangerous to your understanding of the game.

When you're just getting into it, you don't always know what's a hot take designed to farm engagement and what's a real observation. You might read:

> *"Salah is finished, hasn't been world class in years"*

…and think that's just what people believe. But it's not. It's ragebait. Salah is consistently one of the best players in the world.

New fans are the most vulnerable to this because:
- They don't have years of context to filter out bad-faith takes
- Ragebait often *sounds* confident and authoritative
- Engagement algorithms push the most provocative content to the top

This tool gives new fans a second opinion. Paste a tweet and find out: is this something worth thinking about, or is someone just fishing for arguments?

---

## 🎯 What This Project Does

Paste any football tweet → get told instantly whether it's ragebait or a genuine opinion → see exactly **which words** gave it away.

```
Input:  "Arsenal ALWAYS bottle it 💀 name one time they didn't"
Output: ✅ GENUINE — 94.1% confidence
        ↳ "bottle", "didn't" pushed toward genuine — frustrated fan venting, not bait
```

```
Input:  "Really impressed by how Slot set up Liverpool's press today"
Output: ✅ GENUINE — 99.6% confidence
        ↳ "impressed", "setup", "press" all pushed toward genuine
```

---

## 🧠 How It Actually Works

### The Problem With Simple Approaches

You might think — just look for angry words right?

Wrong. Consider:

> *"Messi is not finished"* ← Genuine defence of a player
> *"Messi is finished"* ← Classic ragebait

Same words, opposite meanings. Simple keyword matching fails completely here.

### The Solution — BERTweet + Hybrid Features

This project combines two types of intelligence:

**1. BERTweet (Deep Language Understanding)**

BERTweet is a transformer model pretrained on **850 million real tweets.** It already speaks fluent Football Twitter — it knows what *"cooked"*, *"lmaooo"*, and *"💀"* mean in context before we even start training.

We fine-tuned it on our football ragebait dataset to teach it our specific task. Think of it as hiring someone who already lives on Football Twitter and just teaching them our labeling rules.

**2. Handcrafted Features (Explicit Signal Detection)**

EDA revealed that ragebait tweets have measurably different surface patterns:

| Feature | Genuine | Ragebait |
|---|---|---|
| Caps ratio | 0.020 | 0.031 (+55%) |
| Trigger words | 0.000 | 0.076 |
| Ragebait emojis | rare | common |

So we explicitly extract 7 features per tweet and combine them with BERTweet's 768-dimensional embeddings. Two types of intelligence, one classifier.

**3. Word Importance (Explainability)**

The model doesn't just give you a verdict — it highlights which words drove the decision. We measure this by removing each word and seeing how much the prediction changes. The bigger the drop, the more important the word.

---

## 🏗️ Architecture

```
Raw Tweet
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
BERTweet Encoder                Handcrafted Features
(vinai/bertweet-base)           • caps_ratio
    │                           • exclamation_count
    ▼                           • question_count
CLS Token Embedding             • has_lol/lmao
[768 numbers]                   • has_trigger_words
    │                           • word_count
    │                           • ragebait_emoji
    │                           [7 numbers, StandardScaled]
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
         Combined Vector [775 numbers]
                   │
                   ▼
        Logistic Regression Classifier
                   │
                   ▼
        Genuine (0) or Ragebait (1)
        + Confidence Score
        + Word Importance Explanation
```

---

## 📊 Results

| Metric | Score |
|---|---|
| **F1 Score** | **0.9699** |
| Accuracy | 96.99% |
| Precision | 97.00% |
| Recall | 96.99% |

### Confusion Matrix
```
                 Predicted Genuine   Predicted Ragebait
Actual Genuine        82                    3
Actual Ragebait        2                   79
```

Only **5 mistakes** out of 166 test tweets.

### Fresh Tweet Sanity Check

Tested on 10 completely unseen tweets written after training:

```
✅ "Imagine thinking Haaland is world class lmaooo"     → Ragebait  99.7%
✅ "Arsenal will ALWAYS find a way to bottle it"        → Ragebait  99.7%
✅ "Really impressed by how Slot set up Liverpool"      → Genuine   99.6%
✅ "The tactical battle tonight was fascinating"        → Genuine   99.6%

Score: 10/10 ✅
```

---

## 🔍 Word Importance — The Interesting Part

The model doesn't just classify — it explains itself.

**Clear Ragebait:**
> *"Messi is absolutely finished and anyone defending him is delusional lmaooo 😂"*

🔴 `finished` → strongly ragebait
🔴 `delusional` → strongly ragebait
🔴 `lmaooo` → ragebait
🔴 `😂` → ragebait

**Tricky Genuine:**
> *"Messi had a quiet game but his positioning always creates space"*

🔴 `Messi` → tiny push toward ragebait (appears in provocative tweets often)
🟢 `positioning` → genuine signal
🟢 `creates` → genuine signal
🟢 `quiet` → genuine signal

**Key insight:** The model learned that *"Messi"* alone slightly suggests ragebait because of training data patterns — but surrounding analytical words completely override it. Context over keywords. Every time.

---

## 📁 Project Structure

```
ragebait-detector/
│
├── app.py                  ← Streamlit app (frontend + backend)
├── requirements.txt        ← Dependencies
│
├── models/
│   ├── hybrid_classifier.pkl   ← Trained Logistic Regression
│   └── scaler.pkl              ← StandardScaler for feature normalization
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_bertweet_training.ipynb
│   └── 04_hybrid_model.ipynb
│
├── data/
│   └── dataset_v1.csv      ← 1,103 labeled football tweets
│
└── README.md
```

---

## 📦 Dataset

**1,103 football tweets** across two classes:

| Class | Count |
|---|---|
| Genuine | 566 |
| Ragebait | 537 |

**Collection method:** Synthetically generated with human validation
and manual labeling of 200+ examples by the author following
a documented labeling guide.

**What counts as ragebait:**
- Extreme claims designed to provoke (*"finished", "overrated", "delusional"*)
- Bait comparisons (*"Ronaldo would never"*)
- Sarcastic mockery (*"imagine thinking X lmaooo 😂"*)
- Absolute statements (*"ALWAYS", "NEVER", "worst ever"*)

**What counts as genuine:**
- Match analysis and tactical observations
- Measured player assessments with reasoning
- Statistical discussions
- Historical context and comparisons

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Prasad528260/ragebait-detector.git
cd ragebait-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 🔬 Key Technical Decisions & Why

**Why BERTweet over BERT?**
BERT was trained on Wikipedia and books — clean formal English.
Tweets have slang, emojis, abbreviations, and no grammar.
BERTweet was pretrained on 850M tweets so it already understands
the language domain before fine-tuning even begins.

**Why fine-tune instead of training from scratch?**
Training from scratch needs millions of examples. Fine-tuning
leverages existing pretrained knowledge and only needs hundreds.
With 1,103 samples fine-tuning was the only viable approach.

**Why Logistic Regression over MLP for the hybrid classifier?**
MLP performed slightly worse (F1: 0.9639 vs 0.9699). With ~800
training samples, simpler models generalize better. LR won because
the dataset size didn't justify MLP's additional complexity.

**Why did the hybrid model not improve over BERTweet baseline?**
BERTweet already implicitly captures surface signals (caps, trigger
words, emojis) through deep language understanding. The handcrafted
features didn't add independent information. This is a finding not
a failure — it validates BERTweet's effectiveness for tweet classification.

**Why word importance over SHAP?**
Same core concept — ablation based feature attribution. Measures
prediction change when each word is removed. Avoids library version
conflicts with newer transformers while being equally interpretable.

---

## 📈 What I Learned

This was my first independent NLP project outside of tutorials.
Key lessons:

- **Domain matters more than architecture.** BERTweet's tweet
  pretraining contributed more than any other single decision.
- **EDA informs everything downstream.** Handcrafted features
  came directly from EDA findings — not random choices.
- **Simpler models on small data beat complex ones.** Logistic
  Regression outperformed MLP on 1,103 samples.
- **Explainability builds trust.** Word importance makes the
  model's reasoning transparent and debuggable.
- **Report results honestly.** 0.97 F1 on synthetic data should
  be interpreted carefully. Real world performance on genuinely
  messy tweets would likely be 0.75-0.85 — still strong for a
  niche classification task.

---

## 🛠️ Built With

- [BERTweet](https://huggingface.co/vinai/bertweet-base) — Tweet-specialized transformer
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Model loading and fine-tuning
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [scikit-learn](https://scikit-learn.org/) — Hybrid classifier and evaluation
- [Streamlit](https://streamlit.io/) — Web app and deployment
- [Google Colab](https://colab.research.google.com/) — Training environment (T4 GPU)

---

## 📄 Resume Bullet

> Built a context-aware football ragebait detection system using
> fine-tuned BERTweet and a hybrid NLP pipeline combining
> 768-dimensional transformer embeddings with EDA-validated
> handcrafted features, achieving F1: 0.97 on a custom
> 1,103-tweet dataset with end-to-end deployment via Streamlit.

---

## 👤 Author

**Prasad** — First independent ML project. Built end to end
from data collection to deployment.

⭐ Star this repo if you found it interesting!
