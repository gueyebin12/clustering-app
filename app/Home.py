import streamlit as st

st.set_page_config(page_title="Clustering : K-means et CAH", layout="wide")
st.title("Clustering : K-means et CAH")
st.write("Choisis un jeu de données dans le menu de gauche.")
st.markdown("""
- **Grains de blé** (Seeds, UCI)
- **Vins** (Wine, UCI)
- **Clients de grossiste** (Wholesale customers, UCI)
- **Clients télécom** (K-means, CAH et déploiement du meilleur modèle)
""")
st.caption("Pour chaque jeu : exploration, choix de k, K-means, CAH, comparaison et prédiction d'un nouvel individu.")
