from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE = Path(__file__).parent


@st.cache_data
def load_grains():
    return pd.read_csv(BASE / "data" / "wheat_seeds_dataset.csv"), None


@st.cache_data
def load_wine():
    cols = ["class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
            "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue",
            "OD280/OD315", "Proline"]
    raw = pd.read_csv(BASE / "data" / "wine.data", header=None, names=cols)
    return raw.drop(columns="class"), raw["class"].rename("Cultivar")


@st.cache_data
def load_wholesale():
    raw = pd.read_csv(BASE / "data" / "Wholesale_customers_data.csv")
    return raw.drop(columns=["Channel", "Region"]), raw["Channel"].map({1: "Horeca", 2: "Retail"}).rename("Canal")


@st.cache_resource
def load_model(name):
    return joblib.load(BASE / "models" / name)


def render_clustering_page(title, source_url, loader, model_file, K, ref_label=None):
    """Page complète : données, choix de k, K-means, CAH, comparaison, prédiction."""
    df, ref = loader()
    bundle = load_model(model_file)
    scaler, kmeans, cols = bundle["scaler"], bundle["kmeans"], bundle["columns"]
    use_log = bundle.get("log", False)
    prep = (lambda a: np.log1p(a)) if use_log else (lambda a: a)
    x = scaler.transform(prep(df[cols].values))

    st.title(title)
    st.caption(f"Source : [UCI Machine Learning Repository]({source_url})")
    if use_log:
        st.info("Les dépenses sont très asymétriques : transformation log(1 + x) puis standardisation.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Données", "Choix de k", "K-means", "CAH", "Prédire un cluster"])

    with tab1:
        st.write(f"{df.shape[0]} individus, {df.shape[1]} variables.")
        st.dataframe(df.head(10))
        st.dataframe(df.describe().round(2))

    with tab2:
        ks = list(range(2, 11))
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(x, km.labels_))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Méthode du coude"); ax[0].set_xlabel("k")
        ax[1].plot(ks, sils, "o-"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
        st.pyplot(fig)
        st.write(f"k retenu : **{K}**")

    pca = PCA(n_components=2)
    pcs = pd.DataFrame(pca.fit_transform(x), columns=["PC1", "PC2"])
    km_labels = kmeans.predict(x)

    with tab3:
        st.metric("Silhouette K-means", f"{silhouette_score(x, km_labels):.3f}")
        st.caption(f"Variance expliquée par PC1 + PC2 : {pca.explained_variance_ratio_.sum() * 100:.1f} %")
        st.plotly_chart(px.scatter(pcs, x="PC1", y="PC2", color=km_labels.astype(str),
                                   labels={"color": "Cluster"}, title="Clusters K-means (plan factoriel)"))
        st.write("Moyenne par cluster (valeurs d'origine)")
        st.dataframe(df.assign(cluster=km_labels).groupby("cluster").mean().round(2))

    with tab4:
        Z = linkage(x, method="ward")
        fig, ax = plt.subplots(figsize=(11, 4))
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., show_contracted=True, ax=ax)
        ax.axhline(y=Z[-K, 2] - 1e-6, color="r", linestyle="--", label=f"Coupure à {K} clusters")
        ax.set_title("Dendrogramme (Ward)"); ax.legend()
        st.pyplot(fig)
        cah_labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(x)
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette CAH", f"{silhouette_score(x, cah_labels):.3f}")
        c2.metric("Silhouette K-means", f"{silhouette_score(x, km_labels):.3f}")
        c3.metric("Rand ajusté K-means / CAH", f"{adjusted_rand_score(km_labels, cah_labels):.3f}")
        st.plotly_chart(px.scatter(pcs, x="PC1", y="PC2", color=cah_labels.astype(str),
                                   labels={"color": "Cluster"}, title="Clusters CAH (plan factoriel)"))
        st.write("Tableau croisé K-means / CAH")
        st.dataframe(pd.crosstab(km_labels, cah_labels, rownames=["K-means"], colnames=["CAH"]))
        if ref is not None:
            st.write(f"Comparaison K-means / {ref.name} (non utilisé pour le clustering)")
            st.dataframe(pd.crosstab(km_labels, ref.values, rownames=["K-means"], colnames=[ref.name]))

    with tab5:
        st.write("Saisis les caractéristiques d'un nouvel individu (valeurs d'origine).")
        values = []
        grid = st.columns(3)
        for i, c in enumerate(cols):
            lo, hi, mean = float(df[c].min()), float(df[c].max()), float(df[c].mean())
            values.append(grid[i % 3].number_input(c, min_value=0.0, value=round(mean, 3), format="%.3f"))
        if st.button("Prédire le cluster", key=f"pred_{model_file}"):
            new = scaler.transform(prep(np.array([values])))
            st.success(f"Cet individu appartient au cluster **{int(kmeans.predict(new)[0])}**.")


def render_customer_page():
    """Page 4 : segmentation des clients télécom (K-means, CAH et déploiement CAH par centroïdes)."""
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import normalize

    df = pd.read_csv(BASE / "data" / "Customer_dataset.csv")
    df.columns = [" ".join(c.split()) for c in df.columns]
    bundle = load_model("cah_customer.joblib")
    centroides, classes, noms, cols = bundle["centroides"], bundle["classes"], bundle["noms_classes"], bundle["columns"]
    x = normalize(df[cols].values)

    st.title("Clients d'une entreprise de télécommunications")
    st.caption("Segmentation par K-means et CAH ; le meilleur modèle (CAH) est déployé ci-dessous.")

    tab1, tab2, tab3, tab4 = st.tabs(["Données", "K-means", "CAH", "Affecter un client"])
    pcs = pd.DataFrame(PCA(n_components=2).fit_transform(x), columns=["PC1", "PC2"])

    with tab1:
        st.write(f"{df.shape[0]} clients, {df.shape[1]} variables.")
        st.dataframe(df.head(10))
        st.dataframe(df.describe().T.round(2))

    km = KMeans(n_clusters=3, init="random", max_iter=500, n_init=10, random_state=42).fit(x)
    sil_km = silhouette_score(x, km.labels_)
    with tab2:
        ks = list(range(2, 11))
        inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(x).inertia_ for k in ks]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(ks, inertias, "o-"); ax.set_title("Méthode du coude"); ax.set_xlabel("k")
        st.pyplot(fig)
        st.metric("Silhouette K-means (k = 3)", f"{sil_km:.3f}")
        st.plotly_chart(px.scatter(pcs, x="PC1", y="PC2", color=km.labels_.astype(str), labels={"color": "Classe"}))

    cah_labels = AgglomerativeClustering(n_clusters=2, linkage="average").fit_predict(x)
    sil_cah = silhouette_score(x, cah_labels)
    with tab3:
        Z = linkage(x, "average")
        fig, ax = plt.subplots(figsize=(11, 4))
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90., show_contracted=True, ax=ax)
        ax.set_title("Dendrogramme (lien moyen)")
        st.pyplot(fig)
        c1, c2 = st.columns(2)
        c1.metric("Silhouette CAH (2 classes)", f"{sil_cah:.3f}")
        c2.metric("Silhouette K-means", f"{sil_km:.3f}")
        st.plotly_chart(px.scatter(pcs, x="PC1", y="PC2", color=cah_labels.astype(str), labels={"color": "Classe"}))
        st.write("Profil moyen de chaque classe")
        st.dataframe(df.groupby(cah_labels).mean().round(2).T)
        st.caption("Classe 0 : clients quasi inactifs (non fidèles). Classe 1 : clients actifs (fidèles).")

    with tab4:
        st.write("Saisis les caractéristiques d'un client (pré-remplies avec la médiane).")
        values = []
        grid = st.columns(3)
        for i, c in enumerate(cols):
            values.append(grid[i % 3].number_input(c, value=float(df[c].median()), format="%.2f"))
        if st.button("Affecter le client", key="pred_customer"):
            client = normalize(np.array(values, dtype=float).reshape(1, -1))
            classe = int(classes[cdist(client, centroides).argmin()])
            st.success(f"C'est un client {noms[classe]}.")
