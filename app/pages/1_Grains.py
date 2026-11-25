import streamlit as st
from utils import render_clustering_page, load_grains
st.set_page_config(page_title="Grains", layout="wide")
render_clustering_page("Grains de blé", "https://archive.ics.uci.edu/dataset/236/seeds", load_grains, "kmeans_grains.joblib", K=3)
