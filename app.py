import streamlit as st
import torch
import numpy as np
import pickle
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Football Ragebait Detector",
    page_icon="⚽",
    layout="centered"
)

# ================================================================
# LOAD MODEL — only loads once, cached for performance
# ================================================================
@st.cache_resource
def load_models():
    """
    Cache decorator means this function only runs ONCE
    even if the user interacts with the app multiple times.
    Without caching, model reloads on every button click = very slow.
    """
    # Load BERTweet from Hugging Face directly
    # No need to upload the whole model folder
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base",
        num_labels=2
    )

    # Load your fine tuned weights
    # These are saved in the models/ folder in your repo
    with open("models/hybrid_classifier.pkl", "rb") as f:
        hybrid_clf = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return tokenizer, model, hybrid_clf, scaler, device

# ================================================================
# FEATURE EXTRACTION FUNCTIONS
# ================================================================
def get_bertweet_embedding(text, tokenizer, model, device):
    """
    Convert tweet text into 768 numbers using BERTweet.
    These numbers capture the deep meaning of the tweet.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.base_model(**inputs)
        # CLS token = summary of entire tweet
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    return embedding


def get_handcrafted_features(text):
    """
    Extract 7 explicit signals we discovered during EDA.
    These complement BERTweet's deep understanding
    with surface level ragebait signals.
    """
    text_lower = text.lower()

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    has_lol = int(bool(re.search(r"\blol\b|\blmao\b|\blmaooo\b", text_lower)))
    has_trigger_words = int(bool(re.search(
        r"overrated|finished|delusional|fraud|pathetic|"
        r"embarrassing|cooked|worst|imagine|always|never",
        text_lower
    )))
    word_count = len(text.split())
    has_ragebait_emoji = int(bool(re.search(r"😂|💀|🤣|😭|🙄", text)))

    return np.array([[
        caps_ratio,
        exclamation_count,
        question_count,
        has_lol,
        has_trigger_words,
        word_count,
        has_ragebait_emoji
    ]])


def predict_tweet(text, tokenizer, model, hybrid_clf, scaler, device):
    """
    Full prediction pipeline:
    1. Get BERTweet embedding (768 numbers)
    2. Get handcrafted features (7 numbers)
    3. Scale handcrafted features
    4. Combine both (775 numbers)
    5. Classify with hybrid model
    """
    # Step 1: BERTweet embedding
    embedding = get_bertweet_embedding(text, tokenizer, model, device)

    # Step 2 + 3: Handcrafted features scaled
    hc_features = get_handcrafted_features(text)
    hc_scaled = scaler.transform(hc_features)

    # Step 4: Combine
    combined = np.hstack([embedding, hc_scaled])

    # Step 5: Predict
    prediction = hybrid_clf.predict(combined)[0]
    probability = hybrid_clf.predict_proba(combined)[0]
    confidence = probability[prediction] * 100

    return prediction, confidence


def get_word_importance(text, tokenizer, model, device):
    """
    Measure each word's contribution to the prediction
    by removing it and seeing how much the ragebait
    probability changes. Same concept as SHAP.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        baseline_probs = torch.softmax(
            outputs.logits, dim=1
        ).cpu().numpy()[0]

    baseline_ragebait = baseline_probs[1]
    words = text.split()
    word_importance = []

    for i in range(len(words)):
        masked_tweet = " ".join(words[:i] + words[i+1:])
        if not masked_tweet.strip():
            word_importance.append(0)
            continue

        inputs_masked = tokenizer(
            masked_tweet,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            masked_probs = torch.softmax(
                model(**inputs_masked).logits, dim=1
            ).cpu().numpy()[0]

        word_importance.append(baseline_ragebait - masked_probs[1])

    return words, word_importance

# ================================================================
# UI — HEADER
# ================================================================
st.title("⚽ Football Ragebait Detector")
st.markdown("""
Detect whether a football tweet is **ragebait** or a **genuine opinion**
using a fine-tuned BERTweet model trained on football Twitter.
""")

st.divider()

# ================================================================
# UI — EXAMPLES (helps users understand what to test)
# ================================================================
st.markdown("#### 💡 Try an example or write your own:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Ragebait examples:**")
    if st.button("Messi is finished lmaooo 😂"):
        st.session_state.tweet_input = "Messi is finished lmaooo 😂"
    if st.button("Arsenal ALWAYS bottle it 💀"):
        st.session_state.tweet_input = "Arsenal ALWAYS bottle it 💀"
    if st.button("Ronaldo in Saudi = career over"):
        st.session_state.tweet_input = "Ronaldo in Saudi = career over"

with col2:
    st.markdown("**Genuine examples:**")
    if st.button("Slot's pressing setup was impressive"):
        st.session_state.tweet_input = "Slot's pressing setup was impressive"
    if st.button("Mbappe's movement has improved"):
        st.session_state.tweet_input = "Mbappe's movement has improved"
    if st.button("Tactical battle was fascinating today"):
        st.session_state.tweet_input = "Tactical battle was fascinating today"

# ================================================================
# UI — INPUT BOX
# ================================================================
tweet_input = st.text_area(
    "Or type your own tweet:",
    value=st.session_state.get("tweet_input", ""),
    height=100,
    placeholder="Type any football tweet here..."
)

analyze_button = st.button("🔍 Analyze Tweet", type="primary")

# ================================================================
# UI — PREDICTION OUTPUT
# ================================================================
if analyze_button and tweet_input.strip():

    # Load models (cached after first load)
    with st.spinner("Loading models... (first run takes ~30 seconds)"):
        tokenizer, model, hybrid_clf, scaler, device = load_models()

    with st.spinner("Analyzing tweet..."):
        prediction, confidence = predict_tweet(
            tweet_input, tokenizer, model,
            hybrid_clf, scaler, device
        )

    st.divider()

    # Result display
    if prediction == 1:
        st.error(f"🚨 RAGEBAIT — {confidence:.1f}% confidence")
        st.markdown("""
        This tweet shows signs of **intentional provocation** —
        designed to trigger emotional reactions rather than
        express a genuine football opinion.
        """)
    else:
        st.success(f"✅ GENUINE — {confidence:.1f}% confidence")
        st.markdown("""
        This tweet appears to be a **genuine football opinion** —
        expressing a real view without intentional provocation.
        """)

    # Confidence bar
    st.markdown("#### Confidence Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        genuine_conf = (1 - confidence/100) if prediction == 1 else confidence/100
        st.metric("Genuine", f"{genuine_conf*100:.1f}%")
    with col2:
        ragebait_conf = confidence/100 if prediction == 1 else (1 - confidence/100)
        st.metric("Ragebait", f"{ragebait_conf*100:.1f}%")

    # Word importance section
    st.divider()
    st.markdown("#### 🔍 Why did the model decide this?")
    st.markdown(
        "Each word's contribution to the prediction. "
        "🔴 pushes toward ragebait, 🟢 pushes toward genuine."
    )

    with st.spinner("Calculating word importance..."):
        words, importance = get_word_importance(
            tweet_input, tokenizer, model, device
        )

    # Display word importance as colored text
    word_html = ""
    max_imp = max(abs(v) for v in importance) if importance else 1

    for word, imp in zip(words, importance):
        normalized = imp / max_imp if max_imp > 0 else 0
        if normalized > 0.1:
            # Red — ragebait signal
            intensity = min(int(normalized * 200), 200)
            color = f"rgb(255, {255-intensity}, {255-intensity})"
        elif normalized < -0.1:
            # Green — genuine signal
            intensity = min(int(abs(normalized) * 200), 200)
            color = f"rgb({255-intensity}, 255, {255-intensity})"
        else:
            color = "transparent"

        word_html += f"""
        <span style='background-color: {color};
                     padding: 3px 6px;
                     margin: 3px;
                     border-radius: 4px;
                     font-size: 16px;
                     display: inline-block;'>
            {word}
        </span>
        """

    st.markdown(
        f"<div style='line-height: 2.5;'>{word_html}</div>",
        unsafe_allow_html=True
    )

elif analyze_button and not tweet_input.strip():
    st.warning("Please enter a tweet first!")

# ================================================================
# UI — FOOTER
# ================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 13px;'>
    Built with BERTweet + Hybrid NLP Features<br>
    Fine-tuned on 1,103 football tweets with human validation
</div>
""", unsafe_allow_html=True)