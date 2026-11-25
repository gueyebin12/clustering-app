import streamlit as st
from utils import render_clustering_page, load_wine
st.set_page_config(page_title="Vins", layout="wide")
render_clustering_page("Vins (Wine)", "https://archive.ics.uci.edu/dataset/109/wine", load_wine, "kmeans_wine.joblib", K=3)
