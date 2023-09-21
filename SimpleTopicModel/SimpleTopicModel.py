import warnings
import umap.umap_ as umap
import hdbscan
from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly
import plotly.express as px
from tqdm import tqdm
tqdm.pandas()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return embedder.encode(text, convert_to_tensor=False)

def embed_df(df, colname):
    df[f"{colname}_emb"] = df[colname].progress_apply(lambda x: embed(x))

    return df

def umap_reduction(df, embedding_col, n_components):
    reducer = umap.UMAP(random_state=42, n_components=n_components)
    return reducer.fit_transform(df[embedding_col].tolist())

def train_clusterer(reduced_embeddings, min_cluster_size=None):

    ## Todo: Properly handle how many clusters should come out. Aim for 5-40
    if min_cluster_size == None:
        min_cluster_size = 0.05*len(reduced_embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    clusterer.fit(reduced_embeddings)

    num_clusters = len(set(clusterer.labels_))
    if not 5 < num_clusters < 40:
        warnings.warn(f"{num_clusters} is outside the recommended range of 5-40; consider tuning your min_cluster_size.")

    return clusterer

def add_clusters(df, colname, clusterer):
    df[f'{colname}_cluster'] = clusterer.labels_

    ## Casting the cluster number to a string, so the visuals get interpreted as discrete vals
    df[f'{colname}_cluster'] = df[f'{colname}_cluster'].astype(str)

    return df

def make_embedding_df(reduced_embeddings):

    ## TODO: Smartly handle the new columns based on length of reduced_embeddings[0]
    df_emb = pd.DataFrame(reduced_embeddings).rename(columns={0:'x', 1:'y', 2:'z'})

    return df_emb
def add_reduced_cols(df, df_emb):

    ## Todo: Reduce the douple loop
    for zz in ['x', 'y', 'z']:
        if zz in df.columns:
            df = df.drop(columns=[zz])

    return pd.merge(df, df_emb, on=df.index.values).drop(columns='key_0')


def make_3d_scatter(df, cluster_col, unclustered=False):
    if unclustered == False:
        df = df[df[f'{cluster_col}_cluster'] != "-1"]

    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color=f'{cluster_col}_cluster',
                        hover_data=['title'],
                        width=1000, height=1000
                        )
    fig.update_traces(marker={'size': 3})

    return fig


def prep_for_viz(df, reduced_embeddings):
    df_emb = make_embedding_df(reduced_embeddings)
    df = add_reduced_cols(df, df_emb)

    return df