import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from pymilvus import Collection
from embedder import embed_text
from searcher import perform_search
from utils import prepare_collection

# --- CONFIGURATION ---
DATA_PATH = "data/books.csv"
OLLAMA_URL = "http://ollama:11434/api/generate"

st.set_page_config(page_title="Vector Search Database", page_icon="⚡", layout="wide")
st.title("⚡ Vector Search Database")

# --- 1. SETUP & CACHING ---
@st.cache_resource
def load_embedder():
    return embed_text

embed_text_cached = load_embedder()

if "collection" not in st.session_state:
    st.session_state.collection = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
# State to control showing the visualization
if "show_viz" not in st.session_state:
    st.session_state.show_viz = False

# --- NEW: VISUALIZATION CALCULATION ---
# We cache this data so we don't re-run the heavy math on every click
@st.cache_data
def calculate_visualization_data(collection_name):
    try:
        col = Collection(collection_name)
        col.load()
        # Fetch up to 2000 items to visualize (fetching everything can be slow)
        results = col.query(expr="id >= 0", output_fields=["vector", "meta"], limit=2000)
        if len(results) < 5: return None # Need a few points for t-SNE to work

        # Prepare data
        vectors = np.array([r["vector"] for r in results])
        metas = [r["meta"] for r in results]

        # Run t-SNE (Squash 384 dimensions to 2)
        # Perplexity must be less than number of samples
        perp = min(30, len(vectors) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        X_2d = tsne.fit_transform(vectors)

        # Create DataFrame for Plotly
        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'Description': metas,
            # Simple coloring trick: check if it looks like a book genre string
            'Type': ['Book Genre' if '{' in m else 'Other Media' for m in metas]
        })
        return df
    except Exception as e:
        st.error(f"Viz Error: {e}")
        return None
# ------------------------------------

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Reload Database"):
        with st.spinner("Reloading..."):
            st.session_state.collection = prepare_collection(DATA_PATH)
            st.success("Done!")
            st.cache_data.clear() # Clear cache on reload
    
    use_ai = st.toggle("Enable AI Assistant", value=True)

    st.divider()
    st.header("Visualization")
    # Button to toggle the Galaxy View
    if st.button("🪐 Show/Hide Galaxy View"):
        st.session_state.show_viz = not st.session_state.show_viz
        st.rerun()

# Auto-load DB
if st.session_state.collection is None:
    try:
        st.session_state.collection = prepare_collection(DATA_PATH)
    except: pass

# --- AI FUNCTION ---
def ask_llama(context_list, user_query):
    context_text = "\n".join(context_list)
    prompt = f"""
    You are an expert librarian. User query: "{user_query}"
    Relevant items found:
    {context_text}
    TASK: Recommend the best 1-2 options and explain why they match the query.
    """
    try:
        payload = {"model": "phi3", "prompt": prompt, "stream": False}
        r = requests.post(OLLAMA_URL, json=payload)
        return r.json()['response'] if r.status_code == 200 else "AI Error"
    except: return "AI Offline"

# --- MAIN APP LOGIC ---

# --- 1. VISUALIZATION SECTION (Top) ---
if st.session_state.show_viz and st.session_state.collection:
    st.subheader("Library Map (t-SNE)")
    with st.spinner("Calculating 2D map positions..."):
        viz_df = calculate_visualization_data(st.session_state.collection.name)
        if viz_df is not None:
            # Create interactive scatter plot
            fig = px.scatter(
                viz_df, x='x', y='y',
                hover_data=['Description'], color='Type',
                title="Hover over points to see details",
                template="plotly_dark"
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to visualize yet. Add more books!")
    st.divider()

# --- 2. SEARCH SECTION (Bottom) ---
def run_search_ui(query_text):
    st.session_state.last_query = query_text
    ai_placeholder = st.empty()
    
    query_vec = [embed_text_cached(query_text)]
    results = perform_search(st.session_state.collection, query_vec, limit=5, index_type="HNSW")
    
    if not results:
        st.warning("No matches found.")
        return

    st.subheader(f"Results for: '{query_text}'")
    context_pieces = []
    for i, hit in enumerate(results):
        meta = hit["meta"]
        context_pieces.append(f"- {meta}")
        with st.container():
            c1, c2 = st.columns([0.85, 0.15])
            with c1: st.markdown(f"**{i+1}. {meta}**")
            with c2:
                if st.button("🔍 Similar", key=f"btn_{i}"):
                    st.session_state.last_query = meta
                    st.rerun()
            st.divider()

    if use_ai:
        with ai_placeholder.container():
            with st.status("🧠 AI analyzing...", expanded=True) as status:
                ai_response = ask_llama(context_pieces, query_text)
                st.info(ai_response)
                status.update(label="AI Finished", state="complete", expanded=False)

query_input = st.text_input("Search Collection", value=st.session_state.last_query)
search_btn = st.button("Search")

if search_btn or (st.session_state.last_query and st.session_state.collection):
    if query_input != st.session_state.last_query: st.session_state.last_query = query_input
    if st.session_state.last_query: run_search_ui(st.session_state.last_query)