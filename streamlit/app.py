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

# ---------------------------------------------------------------- graphiques de l'onglet « Analyses du notebook »
def figure_log(rfm):
    """Question 1 : distributions avant et après log1p."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for j, var in enumerate(VARIABLES):
        brut = rfm[var]
        log = np.log1p(rfm[var])
        axes[0, j].hist(brut, bins=40, color='#9ecae1')
        axes[0, j].set_title(f"{var} (brut) - asymétrie {brut.skew():.2f}")
        axes[1, j].hist(log, bins=40, color='#3182bd')
        axes[1, j].set_title(f"{var} (log1p) - asymétrie {log.skew():.2f}")
    fig.tight_layout()
    return fig


def figure_correlations(rfm):
    """Question 2 : corrélations entre R, F et M après log1p."""
    corr = np.log1p(rfm[VARIABLES]).corr()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES)
    ax.set_yticks(range(len(VARIABLES)))
    ax.set_yticklabels(VARIABLES)
    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center')
    fig.colorbar(image)
    ax.set_title('Corrélations entre R, F et M (après log1p)')
    return fig


def figure_pareto(rfm):
    """Question 3 : concentration du chiffre d'affaires."""
    ca = rfm['Monetary'].sort_values(ascending=False).reset_index(drop=True)
    part_clients = (np.arange(1, len(ca) + 1) / len(ca)) * 100
    part_ca = ca.cumsum() / ca.sum() * 100
    top20 = part_ca[int(0.2 * len(ca)) - 1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(part_clients, part_ca, label='Courbe de concentration')
    ax.plot([0, 100], [0, 100], '--', color='grey', label='Répartition égale')
    ax.axvline(20, color='red', linestyle=':')
    ax.axhline(top20, color='red', linestyle=':')
    ax.set_xlabel('% des clients (classés du plus gros au plus petit)')
    ax.set_ylabel("% du chiffre d'affaires cumulé")
    ax.set_title(f"Les 20 % meilleurs clients font {top20:.0f} % du CA")
    ax.legend()
    return fig


@st.cache_data
def evaluer_k(valeurs_k=tuple(range(2, 11))):
    """Question 4 : inertie et silhouette pour plusieurs valeurs de K."""
    from sklearn.metrics import silhouette_score
    modele = construire_modele()
    X = modele['scaler'].transform(np.log1p(modele['data'][VARIABLES]))
    lignes = []
    for k in valeurs_k:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        lignes.append({'K': k, 'Inertie': km.inertia_, 'Silhouette': silhouette_score(X, km.labels_)})
    return pd.DataFrame(lignes)


def figure_choix_k(evaluation):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(evaluation['K'], evaluation['Inertie'], marker='o')
    axes[0].set_title("Méthode de l'Elbow")
    axes[0].set_xlabel('Nombre de clusters (K)')
    axes[0].set_ylabel('Inertie')
    axes[1].plot(evaluation['K'], evaluation['Silhouette'], marker='o', color='green')
    axes[1].set_title('Score de silhouette')
    axes[1].set_xlabel('Nombre de clusters (K)')
    axes[1].set_ylabel('Silhouette')
    for ax in axes:
        ax.axvline(K, color='red', linestyle=':', label=f'K retenu = {K}')
        ax.legend()
    fig.tight_layout()
    return fig


def figure_profils(modele):
    """Question 5 : profil moyen des segments (variables standardisées)."""
    donnees = modele['data']
    z = pd.DataFrame(modele['scaler'].transform(np.log1p(donnees[VARIABLES])), columns=VARIABLES)
    z['Segment'] = donnees['Segment'].values
    profil = z.groupby('Segment')[VARIABLES].mean().loc[list(COULEURS)]
    limite = float(np.abs(profil.values).max())

    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(profil, cmap='coolwarm', vmin=-limite, vmax=limite, aspect='auto')
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES)
    ax.set_yticks(range(len(profil)))
    ax.set_yticklabels(profil.index)
    for i in range(len(profil)):
        for j in range(len(VARIABLES)):
            ax.text(j, i, f"{profil.iloc[i, j]:.2f}", ha='center', va='center')
    fig.colorbar(image)
    ax.set_title('Profil moyen des segments (variables standardisées)')
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

onglet_exploration, onglet_qualification, onglet_notebook = st.tabs(
    ["📊 Exploration", "🎯 Qualification client", "📚 Analyses du notebook"]
)

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

            # ------------------------------------------------ onglet 3 : analyses du notebook
with onglet_notebook:
    st.header("Analyses du notebook")
    st.write("Les cinq questions d'analyse du notebook, avec leurs graphiques et leurs commentaires.")

    # Question 1
    st.subheader("Question 1 : pourquoi faut-il transformer les variables avec log1p ?")
    fig = figure_log(donnees)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Commentaire :** ce graphique compare les distributions avant (ligne du haut) et après (ligne du bas) la transformation `log1p`. Nous voulons démontrer que la fréquence et surtout le montant sont très asymétriques : la plupart des clients dépensent peu, et quelques-uns dépensent énormément. Sans transformation, ces valeurs extrêmes domineraient le calcul des distances et fausseraient K-means. Après `log1p`, les distributions sont beaucoup plus équilibrées, ce qui justifie cette étape de prétraitement.")

    # Question 2
    st.subheader("Question 2 : les trois variables RFM apportent-elles des informations différentes ?")
    fig = figure_correlations(donnees)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Commentaire :** nous cherchons à savoir si certaines variables sont redondantes avant le clustering. La fréquence et le montant sont fortement corrélés : un client qui commande souvent dépense logiquement davantage. La récence est corrélée négativement aux deux : les clients récents sont aussi les plus actifs. Nous conservons néanmoins les trois variables, car elles décrivent trois dimensions complémentaires du comportement d'achat.")

    # Question 3
    st.subheader("Question 3 : le chiffre d'affaires dépend-il d'un petit nombre de clients ?")
    fig = figure_pareto(donnees)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Commentaire :** cette courbe de concentration (type Pareto) montre quelle part du chiffre d'affaires est générée par les meilleurs clients. Plus la courbe s'éloigne de la diagonale (répartition égale), plus l'activité est concentrée. Nous voulons démontrer que tous les clients n'ont pas la même valeur, ce qui justifie la segmentation : les actions marketing doivent être différentes selon les groupes.")

    # Question 4
    st.subheader("Question 4 : combien de segments de clients distincts existe-t-il ?")
    evaluation = evaluer_k()
    fig = figure_choix_k(evaluation)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(evaluation.round(3))
    st.markdown("**Commentaire :** nous cherchons le nombre de clusters à retenir avec deux critères complémentaires : l'inertie (le coude de la courbe) et le score de silhouette (qualité de la séparation des clusters). Ici, le coude est progressif et peu net. La silhouette est la plus élevée pour K = 2, mais cette solution sépare seulement les clients actifs des clients inactifs, ce qui est trop grossier pour un usage marketing. Nous retenons donc K = 4, un compromis entre qualité statistique et segments interprétables.")

    # Question 5
    st.subheader("Question 5 : quels sont les profils des segments et leur poids économique ?")
    fig = figure_profils(modele)
    st.pyplot(fig)
    plt.close(fig)
    resume = donnees.groupby('Segment').agg(
        Clients=('CustomerID', 'count'),
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
        CA_total=('Monetary', 'sum'),
    ).loc[list(COULEURS)]
    resume['% clients'] = resume['Clients'] / resume['Clients'].sum() * 100
    resume['% CA'] = resume['CA_total'] / resume['CA_total'].sum() * 100
    st.dataframe(resume.drop(columns='CA_total').round(1))
    st.markdown("**Commentaire :** la heatmap montre, pour chaque segment, si ses clients sont au-dessus (rouge) ou en dessous (bleu) de la moyenne sur chaque variable, ce qui permet de nommer les segments. Les **meilleurs clients** sont peu nombreux mais font l'essentiel du chiffre d'affaires, alors que les **clients perdus** sont nombreux pour un poids économique très faible. Chaque segment appelle donc une action marketing différente : fidéliser les meilleurs clients, développer les réguliers, activer les nouveaux et relancer les inactifs.")