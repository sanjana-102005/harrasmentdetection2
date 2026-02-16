import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="SafeGuard AI - Incident Reporter", page_icon="🛡️", layout="centered")

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    .support-box { padding: 20px; border-radius: 10px; background-color: #e1f5fe; border-left: 5px solid #03a9f4; }
    .legal-box { padding: 20px; border-radius: 10px; background-color: #fff3e0; border-left: 5px solid #ff9800; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL TRAINING & CACHING ---
@st.cache_resource
def train_and_load_model():
    """Trains the model using the real train.csv provided."""
    try:
        data = pd.read_csv('train.csv')
        # Create a single 'harassment' label from the multi-labels in the CSV
        target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        data['is_harassment'] = (data[target_cols].sum(axis=1) > 0).astype(int)
        
        # We use a Pipeline for clean deployment
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))),
            ('clf', LogisticRegression(class_weight='balanced'))
        ])
        
        model_pipeline.fit(data['comment_text'], data['is_harassment'])
        return model_pipeline
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None

# --- HYBRID CLASSIFICATION LOGIC ---
def classify_incident(text):
    text_lower = text.lower()
    
    # Rule-Based Categories & Indian Legal Mapping
    rules = {
        "Sexual Harassment": {
            "keywords": [r"touch", r"sexual", r"comment", r"body", r"unwanted", r"harassment", r"demanded"],
            "legal": "Section 354A IPC (BNS Section 74/75)",
            "tips": "Save any inappropriate messages or CCTV footage if available."
        },
        "Cyber/Stalking": {
            "keywords": [r"online", r"following", r"watching", r"hacked", r"social media", r"profile", r"repeatedly"],
            "legal": "Section 354D IPC (BNS Section 78) & IT Act 66E",
            "tips": "Take screenshots immediately. Do not delete the chat history."
        },
        "Workplace Harassment": {
            "keywords": [r"boss", r"promotion", r"office", r"colleague", r"salary", r"fired", r"cabin"],
            "legal": "POSH Act 2013 & Section 354 IPC",
            "tips": "Report to your Internal Complaints Committee (ICC) within 3 months."
        },
        "Threats & Intimidation": {
            "keywords": [r"kill", r"hit", r"beat", r"hurt", r"attack", r"force", r"consequences"],
            "legal": "Section 503/506 IPC (BNS Section 351)",
            "tips": "Identify witnesses and record audio if safe to do so."
        }
    }
    
    matched_cats = []
    for cat, info in rules.items():
        if any(re.search(kw, text_lower) for kw in info['keywords']):
            matched_cats.append((cat, info))
            
    return matched_cats

# --- MAIN APP UI ---
def main():
    st.title("🛡️ SafeGuard AI")
    st.subheader("Gender-Inclusive Harassment Detection & Support")
    st.write("A secure space to describe your experience and receive guidance.")

    model = train_and_load_model()
    
    incident_text = st.text_area(
        "Describe what happened in your own words:",
        placeholder="E.g., Someone is sending me threatening messages on Instagram...",
        height=150
    )

    if st.button("Analyze & Get Help"):
        if not incident_text or len(incident_text) < 10:
            st.warning("Please provide a more detailed description for an accurate analysis.")
            return

        # 1. ML Prediction (Confidence Score)
        prob = model.predict_proba([incident_text])[0][1]
        
        # 2. Rule-based categorization
        categories = classify_incident(incident_text)
        
        st.divider()

        # CASE: HARASSMENT DETECTED (High Confidence or Rule Match)
        if prob > 0.45 or len(categories) > 0:
            st.error(f"### Assessment: Harassment/Safety Risk Detected")
            st.write(f"**AI Confidence Score:** {prob*100:.1f}%")
            
            # Display Legal Guidance
            st.markdown("### ⚖️ Legal & Safety Guidance (India)")
            if categories:
                for cat, info in categories:
                    st.markdown(f"""
                    <div class="legal-box">
                        <strong>Category: {cat}</strong><br>
                        <strong>Relevant Law:</strong> {info['legal']}<br>
                        <strong>Next Steps:</strong> {info['tips']}
                    </div><br>
                    """, unsafe_allow_html=True)
            else:
                st.info("General Harassment detected. Recommended action: File a report at your local police station under general intimidation laws.")

            # SOS Contacts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🚨 **Emergency Contacts**")
                st.write("- Women Helpline: 1091 / 181")
                st.write("- National Police: 112")
            with col2:
                st.markdown("🌐 **Online Reporting**")
                st.write("- [National Cyber Crime Portal](https://cybercrime.gov.in)")
                st.write("- [NCW Complaint Link](http://ncwapps.nic.in)")

        # CASE: LOW RISK / MILD INCIDENT
        else:
            st.success("### Assessment: Low Immediate Threat Level")
            st.write("Based on the current input, the system does not detect severe harassment or immediate legal threat.")
            
            st.markdown(f"""
            <div class="support-box">
                <h4>🌱 Support & Wellness</h4>
                <p>It sounds like you've had a difficult experience. Even if it doesn't meet the legal threshold for harassment, your feelings are valid.</p>
                <ul>
                    <li>Take a moment for <strong>Box Breathing</strong> (Inhale 4s, Hold 4s, Exhale 4s).</li>
                    <li>Talk to a trusted friend or mentor about this interaction.</li>
                    <li>Keep a log of these incidents in case they become a pattern.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.title("Privacy Info")
    st.sidebar.info("This application processes your input locally within the session. We do not store your incident descriptions on our servers.")

if __name__ == "__main__":
    main()
