"""Dashboard Streamlit : segmentation RFM des clients de GlobalShop Direct."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="GlobalShop Direct - Segmentation RFM", page_icon="🛒", layout="wide")

VARIABLES = ['Recency', 'Frequency', 'Monetary']
K = 4
COULEURS = {
    'Meilleurs clients': '#2ca02c',
    'Clients réguliers': '#1f77b4',
    'Récents peu actifs': '#ff7f0e',
    'Clients perdus': '#d62728',
}
PERSONAS = {
    'Meilleurs clients': {
        'persona': 'Les Fidèles VIP',
        'profil': 'Achètent très souvent, récemment et pour des montants élevés.',
        'action': 'Programme VIP, avantages exclusifs, accès anticipé aux nouveautés.',
    },
    'Clients réguliers': {
        'persona': 'Les Réguliers',
        'profil': 'Achètent régulièrement, avec un dernier achat de quelques semaines.',
        'action': 'Offres de montée en gamme et recommandations personnalisées.',
    },
    'Récents peu actifs': {
        'persona': 'Les Nouveaux à activer',
        'profil': 'Ont acheté récemment mais ne sont revenus que peu de fois.',
        'action': "Email de bienvenue, offre sur la 2e ou 3e commande pour installer l'habitude.",
    },
    'Clients perdus': {
        'persona': 'Les Endormis',
        'profil': "N'ont pas acheté depuis plusieurs mois, souvent une seule commande.",
        'action': 'Campagne de réactivation avec remise ciblée, avant de les considérer comme perdus.',
    },
}


# ---------------------------------------------------------------- données et modèle
@st.cache_data
def charger_donnees(chemin=Path(__file__).parent / 'rfm.csv'):
    return pd.read_csv(chemin)


@st.cache_resource
def construire_modele():
    """Reproduit la méthode du notebook : log1p, standardisation, K-means, PCA."""
    rfm = charger_donnees().copy()

    rfm_log = np.log1p(rfm[VARIABLES])
    scaler = StandardScaler().fit(rfm_log)
    X = pd.DataFrame(scaler.transform(rfm_log), columns=VARIABLES)

    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10).fit(X)
    rfm['Cluster'] = kmeans.labels_

    # Les numéros de cluster sont arbitraires : on nomme les segments d'après leur profil
    profils = rfm.groupby('Cluster')[VARIABLES].mean()
    noms = {}
    noms[profils['Monetary'].idxmax()] = 'Meilleurs clients'
    noms[profils['Recency'].idxmax()] = 'Clients perdus'
    restants = sorted([c for c in profils.index if c not in noms], key=lambda c: profils.loc[c, 'Monetary'])
    noms[restants[0]] = 'Récents peu actifs'
    noms[restants[1]] = 'Clients réguliers'
    rfm['Segment'] = rfm['Cluster'].map(noms)

    pca = PCA(n_components=2, random_state=0).fit(X)
    coords = pca.transform(X)
    rfm['PC1'], rfm['PC2'] = coords[:, 0], coords[:, 1]

    return {'data': rfm, 'scaler': scaler, 'kmeans': kmeans, 'pca': pca, 'noms': noms}


def qualifier_client(modele, recence, frequence, montant):
    """Assigne un nouveau client à un segment à partir de ses valeurs RFM brutes."""
    x = pd.DataFrame([[recence, frequence, montant]], columns=VARIABLES)
    x_scaled = pd.DataFrame(modele['scaler'].transform(np.log1p(x)), columns=VARIABLES)
    cluster = int(modele['kmeans'].predict(x_scaled)[0])
    coords = modele['pca'].transform(x_scaled)[0]
    distances = modele['kmeans'].transform(x_scaled)[0]
    distances_segments = {modele['noms'][i]: float(d) for i, d in enumerate(distances)}
    return modele['noms'][cluster], coords, distances_segments


# ---------------------------------------------------------------- graphiques
def formater_nombre(valeur, decimales=0):
    return f"{valeur:,.{decimales}f}".replace(',', ' ')


def figure_pca(donnees, pca, point=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for segment, couleur in COULEURS.items():
        sous = donnees[donnees['Segment'] == segment]
        ax.scatter(sous['PC1'], sous['PC2'], s=14, alpha=0.6, color=couleur, label=segment)
    if point is not None:
        ax.scatter(point[0], point[1], s=350, marker='*', color='black', edgecolor='white', linewidth=1.2,
                   label='Client saisi', zorder=5)
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f} % de variance)")
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f} % de variance)")
    ax.set_title('Segments projetés en 2D (PCA)')
    ax.legend()
    return fig


def figure_repartition(donnees):
    resume = donnees.groupby('Segment').agg(Clients=('CustomerID', 'count'), CA=('Monetary', 'sum'))
    resume['% clients'] = resume['Clients'] / resume['Clients'].sum() * 100
    resume['% CA'] = resume['CA'] / resume['CA'].sum() * 100
    ordre = [s for s in COULEURS if s in resume.index]
    resume = resume.loc[ordre]

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(resume))
    ax.bar(positions - 0.2, resume['% clients'], width=0.4, label='% des clients', color='#9ecae1')
    ax.bar(positions + 0.2, resume['% CA'], width=0.4, label="% du chiffre d'affaires", color='#3182bd')
    ax.set_xticks(positions)
    ax.set_xticklabels(resume.index, rotation=15)
    ax.set_ylabel('Pourcentage')
    ax.set_title("Répartition des clusters : clients et chiffre d'affaires")
    ax.legend()
    return fig


# ---------------------------------------------------------------- interface
modele = construire_modele()
donnees = modele['data']

st.title("🛒 GlobalShop Direct : segmentation RFM des clients")
st.caption("Dashboard d'exploration des segments clients et outil de qualification en direct (K-means, K = 4).")

# Filtre de la barre latérale
st.sidebar.header("Filtres")
segments_choisis = st.sidebar.multiselect("Segments affichés", options=list(COULEURS), default=list(COULEURS))
st.sidebar.caption("Le filtre s'applique à l'onglet « Exploration ».")

onglet_exploration, onglet_qualification = st.tabs(["📊 Exploration", "🎯 Qualification client"])

# ------------------------------------------------ onglet 1 : exploration
with onglet_exploration:
    filtre = donnees[donnees['Segment'].isin(segments_choisis)]
    if filtre.empty:
        st.warning("Sélectionne au moins un segment dans la barre latérale.")
    else:
        # Cartes KPI
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clients", formater_nombre(len(filtre)))
        col2.metric("Chiffre d'affaires", f"{formater_nombre(filtre['Monetary'].sum())} £")
        col3.metric("Montant médian par client", f"{formater_nombre(filtre['Monetary'].median())} £")
        col4.metric("Récence médiane", f"{formater_nombre(filtre['Recency'].median())} jours")

        # Graphiques
        gauche, droite = st.columns(2)
        with gauche:
            st.subheader("Répartition des clusters")
            fig = figure_repartition(filtre)
            st.pyplot(fig)
            plt.close(fig)
        with droite:
            st.subheader("Projection PCA en 2D")
            fig = figure_pca(filtre, modele['pca'])
            st.pyplot(fig)
            plt.close(fig)

        # Tableau des moyennes et médianes
        st.subheader("Moyennes et médianes RFM par segment")
        stats = filtre.groupby('Segment')[VARIABLES].agg(['mean', 'median']).round(1)
        stats = stats.loc[[s for s in COULEURS if s in stats.index]]
        st.dataframe(stats)

        # Personas
        st.subheader("Personas marketing")
        for segment in [s for s in COULEURS if s in segments_choisis]:
            infos = PERSONAS[segment]
            with st.expander(f"{infos['persona']} ({segment})"):
                st.write(f"**Profil :** {infos['profil']}")
                st.write(f"**Action marketing :** {infos['action']}")

# ------------------------------------------------ onglet 2 : qualification client
with onglet_qualification:
    st.subheader("Qualifier un client en direct")
    st.write("Saisis les valeurs RFM d'un client : le modèle lui assigne automatiquement un segment.")

    with st.form("formulaire_rfm"):
        c1, c2, c3 = st.columns(3)
        recence = c1.number_input("Récence (jours depuis le dernier achat)", min_value=0, value=30, step=1)
        frequence = c2.number_input("Fréquence (nombre de commandes)", min_value=1, value=3, step=1)
        montant = c3.number_input("Montant total dépensé (£)", min_value=0.0, value=500.0, step=10.0)
        envoye = st.form_submit_button("Qualifier ce client")

    if envoye:
        segment, coords, distances = qualifier_client(modele, recence, frequence, montant)
        infos = PERSONAS[segment]

        st.success(f"Segment assigné : **{segment}** (persona : {infos['persona']})")

        hors_plage = [
            nom for nom, valeur in zip(VARIABLES, [recence, frequence, montant])
            if valeur > donnees[nom].max()
        ]
        if hors_plage:
            st.warning("Valeur supérieure au maximum observé dans la base pour : " + ", ".join(hors_plage)
                       + ". L'assignation peut être moins fiable.")

        gauche, droite = st.columns(2)
        with gauche:
            st.write(f"**Profil :** {infos['profil']}")
            st.write(f"**Action marketing recommandée :** {infos['action']}")

            st.write("**Comparaison avec les médianes du segment**")
            mediane = donnees[donnees['Segment'] == segment][VARIABLES].median()
            comparaison = pd.DataFrame({
                'Client saisi': [recence, frequence, montant],
                'Médiane du segment': mediane.round(1).tolist(),
            }, index=['Récence (jours)', 'Fréquence (commandes)', 'Montant (£)'])
            st.dataframe(comparaison)

            st.write("**Distance aux centres des segments** (plus elle est faible, plus le client est proche)")
            tableau_distances = pd.DataFrame({'Distance': distances}).sort_values('Distance').round(2)
            st.dataframe(tableau_distances)
        with droite:
            fig = figure_pca(donnees, modele['pca'], point=coords)
            st.pyplot(fig)
            plt.close(fig)