import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sns.set_style("whitegrid")

# ---------- Data Loader ----------
@st.cache_data
def load_data(path="netflix_cleaned.csv"):
    df = pd.read_csv(path)
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float, errors='ignore')
    df['duration_unit'] = df['duration'].str.extract(r'([a-zA-Z]+)')
    df['listed_in'] = df['listed_in'].fillna('')
    return df

# ---------- Recommender ----------
@st.cache_data
def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['listed_in'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_title(df, sim_matrix, title, n=5):
    if title not in df['title'].values:
        return None
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    recs = [df.iloc[i[0]]['title'] for i in sim_scores]
    return recs

# ---------- Trend Plot ----------
def plot_trend(df):
    yearly = df['year_added'].value_counts().sort_index().dropna()
    X = np.array(yearly.index).reshape(-1,1)
    y = np.array(yearly.values)

    model = LinearRegression()
    model.fit(X, y)
    future_years = np.arange(yearly.index.min(), yearly.index.max()+5).reshape(-1,1)
    preds = model.predict(future_years)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(yearly.index, y, marker='o', label="Actual", color="blue")
    ax.plot(future_years.flatten(), preds, color='red', linestyle='--', label="Predicted Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Titles Added")
    ax.set_title("📈 Netflix Content Trend Prediction")
    ax.legend()
    return fig

# ---------- Main ----------
def main():
    st.set_page_config(page_title="Netflix Analysis & Recommender", layout="wide")
    st.title("🎬 Netflix Analysis & Recommendation App")

    # Load data
    df = load_data()

    # Sidebar Filters
    st.sidebar.header("🔎 Filters")
    sel_type = st.sidebar.multiselect("Type", df['type'].unique(), default=df['type'].unique())
    sel_country = st.sidebar.multiselect("Country", df['country'].dropna().unique())

    filtered = df.copy()
    if sel_type:
        filtered = filtered[filtered['type'].isin(sel_type)]
    if sel_country:
        filtered = filtered[filtered['country'].isin(sel_country)]

    # Quick Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Titles", len(filtered))
    c2.metric("Movies", (filtered['type'] == "Movie").sum())
    c3.metric("TV Shows", (filtered['type'] == "TV Show").sum())

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Type Distribution", "📈 Trends", "🎭 Genres", "🎥 Recommender"])

    with tab1:
        st.subheader("Movies vs TV Shows")
        type_counts = filtered['type'].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=type_counts.index, y=type_counts.values, palette="Set2", ax=ax1)
        ax1.set_xlabel("Type")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Content Added Over Years")
        fig2 = plot_trend(filtered)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Top 10 Genres by Type")
        df_expl = filtered.copy()
        df_expl['genre'] = df_expl['listed_in'].str.split(', ')
        df_expl = df_expl.explode('genre')
        genre_type_counts = df_expl.groupby(['genre', 'type']).size().reset_index(name='count')
        top_genres = df_expl['genre'].value_counts().nlargest(10).index
        genre_type_counts = genre_type_counts[genre_type_counts['genre'].isin(top_genres)]

        fig3, ax3 = plt.subplots(figsize=(10,5))
        sns.barplot(data=genre_type_counts, x='genre', y='count', hue='type', palette="Set2", ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig3)

    with tab4:
        st.subheader("Recommendation System (Genre-Based)")

        sim_matrix = build_recommender(df)
        all_titles = sorted(df['title'].dropna().unique())

        # Searchable dropdown
        title_input = st.selectbox("Choose a Netflix title:", all_titles, index=0)
        rec_count = st.slider("How many recommendations?", min_value=1, max_value=10, value=5)

        if st.button("Recommend 🎥"):
            recs = recommend_title(df, sim_matrix, title_input, n=rec_count)
            if recs:
                for i, title in enumerate(recs, start=1):
                    st.write(f"**{i}. {title}**")
            else:
                st.warning("No recommendations found.")

if __name__ == "__main__":
    main()
