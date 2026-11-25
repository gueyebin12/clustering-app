import streamlit as st
from utils import render_clustering_page, load_wholesale
st.set_page_config(page_title="Clients de grossiste", layout="wide")
render_clustering_page("Clients de grossiste (Wholesale)", "https://archive.ics.uci.edu/dataset/292/wholesale+customers", load_wholesale, "kmeans_wholesale.joblib", K=2)
