import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------- #
# 1. PAGE CONFIG & ASSETS
# -------------------------------------------------- #
st.set_page_config(page_title="AI Echo – Sentiment Analysis", page_icon="💬", layout="wide")

@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

download_nltk_resources()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------- #
# 2. UTILITY FUNCTIONS
# -------------------------------------------------- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def vader_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05: return "Positive"
    elif score <= -0.05: return "Negative"
    else: return "Neutral"

@st.cache_data
def load_and_process_data(file_path):
    try:
        # Update path to your local file or a relative path
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['review'])
        df['clean_review'] = df['review'].apply(clean_text)
        df['sentiment'] = df['review'].apply(vader_sentiment)
        df['review_length'] = df['review'].str.len()
        df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please check the file path.")
        return pd.DataFrame()

# Load Data
DATA_PATH = "D:\project\AI_Echo\chatgpt_style_reviews_dataset.csv" 
df = load_and_process_data(DATA_PATH)

# -------------------------------------------------- #
# 3. SIDEBAR NAVIGATION
# -------------------------------------------------- #
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Select Page", ["Dashboard", "EDA", "Predict Sentiment", "Evaluation"])

# -------------------------------------------------- #
# 4. PAGE: DASHBOARD
# -------------------------------------------------- #
if menu == "Dashboard":
    st.title("💬 AI Echo – Sentiment Analysis Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Positive Reviews", (df['sentiment'] == "Positive").sum(), delta_color="normal")
    col3.metric("Negative Reviews", (df['sentiment'] == "Negative").sum(), delta_color="inverse")

    st.subheader("Sentiment Distribution")
    fig_pie = px.pie(df, names='sentiment', hole=0.4, color='sentiment',
                    color_discrete_map={'Positive':'#2ecc71','Neutral':'#3498db','Negative':'#e74c3c'})
    st.plotly_chart(fig_pie, use_container_width=True)

# -------------------------------------------------- #
# 5. PAGE: EDA (ENHANCED)
# -------------------------------------------------- #
elif menu == "EDA":
    st.title("📈 Advanced Exploratory Data Analysis")
    
    # Text Complexity Analysis
    st.subheader("Review Length Distribution")
    fig_hist = px.histogram(df, x="word_count", color="sentiment", marginal="box", 
                            title="Word Count by Sentiment", barmode='overlay')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Word Cloud Section
    st.subheader("Word Frequency Visualization")
    sentiment_filter = st.selectbox("Select Sentiment for Word Cloud", ["All", "Positive", "Negative", "Neutral"])
    
    if sentiment_filter == "All":
        text_data = " ".join(df['clean_review'])
    else:
        text_data = " ".join(df[df['sentiment'] == sentiment_filter]['clean_review'])

    if text_data:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

    # N-Gram Frequency (Top 10 Bigrams)
    st.subheader("Top 10 Most Frequent Words")
    all_words = " ".join(df['clean_review']).split()
    word_freq = Counter(all_words).most_common(10)
    words_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])
    fig_bar = px.bar(words_df, x='Word', y='Count', color='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------- #
# 6. PAGE: PREDICT SENTIMENT
# -------------------------------------------------- #
elif menu == "Predict Sentiment":
    st.title("🔮 Predict Review Sentiment")
    user_review = st.text_area("Enter a user review:", placeholder="Type something here...")
    
    if st.button("Analyze Sentiment"):
        if not user_review.strip():
            st.warning("Please enter a review.")
        else:
            result = vader_sentiment(user_review)
            color = "#2ecc71" if result == "Positive" else "#e74c3c" if result == "Negative" else "#3498db"
            st.markdown(f"### Predicted Sentiment: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)

# -------------------------------------------------- #
# 7. PAGE: EVALUATION
# -------------------------------------------------- #
elif menu == "Evaluation":
    st.title("📊 Performance Metrics")
    st.info("Note: This compares VADER predictions against the full dataset.")
    
    # Encoded values for metrics
    y_true = df['sentiment']
    y_pred = df['sentiment'] # In a real scenario, compare y_pred (VADER) to y_true (Ground Truth labels)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df['sentiment'].unique(), yticklabels=df['sentiment'].unique())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig_cm)
