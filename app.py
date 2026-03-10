import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Page Configuration
st.set_page_config(page_title="Book Matchmaker", layout="wide")

# --- 1. Load & Process Data ---
@st.cache_data
def load_data():
    # Loading your uploaded books.csv
    df = pd.read_csv('books.csv')
    df['authors'] = df['authors'].fillna('')
    # We use title and authors to find matches
    df['content'] = df['title'] + " " + df['authors']
    return df

df = load_data()

# --- 2. Build the Engine ---
@st.cache_resource # Cache the matrix so it doesn't recalculate on every toggle
def build_engine(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['content'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

cosine_sim = build_engine(df)

# --- 3. Streamlit UI ---
st.title("📚 Book Matchmaker")
st.write("Based on the Goodreads-10k dataset")

# Search/Dropdown Box
book_list = df['title'].values
selected_book = st.selectbox("Search for a book you've read:", book_list)

if st.button('Find Similar Books'):
    # Get index of the selected book
    idx = df[df['title'] == selected_book].index[0]
    
    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6] # Top 5
    
    st.markdown(f"### Because you liked **{selected_book}**...")
    
    # Create 5 columns for the 5 recommendations
    cols = st.columns(5)
    
    for i, (index, score) in enumerate(sim_scores):
        with cols[i]:
            book_title = df['title'].iloc[index]
            # Use the image_url column from your CSV
            img_url = df['image_url'].iloc[index]
            
            st.image(img_url, use_column_width=True)
            # Shorten title if it's too long for the UI
            display_title = (book_title[:30] + '..') if len(book_title) > 30 else book_title
            st.caption(f"**{display_title}**")