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
    """
    Generates a MiniLM embedding from raw text.

    Args:
        text (str): Text to be semantically embedded
    Returns:
        list: MiniLM Embedding of the input text
    """

    return embedder.encode(text, convert_to_tensor=False)

def embed_df(df, colname):
    """
    Adds an embedding Series to a Pandas DataFrame.

    Args:
        df (Pandas DataFrame): Pandas Dataframe with text data
        colname (str): Name of column to embed (for later use clustering)

    Returns:
        DataFrame: Pandas dataframe with new column of semantic embeddings
    """
    df[f"{colname}_emb"] = df[colname].progress_apply(lambda x: embed(x))

    return df

def umap_reduction(df, embedding_col, n_components=3):
    """

    Args:
        df (Pandas DataFrame): Input dataframe with embeddings
        embedding_col (str): Column in dataframe with MiniLM embeddings
        n_components (int): Desired number of dimensions to reduce the embeddings to.

    Returns:
        Array: Numpy matrix of dimension-reduced embeddings
    """

    reducer = umap.UMAP(random_state=42, n_components=n_components)
    return reducer.fit_transform(df[embedding_col].tolist())

def train_clusterer(reduced_embeddings, min_cluster_size=None):
    """
    Trains an HDBSCAN Clusterer on embeddings.

    Args:
        reduced_embeddings: UMAP-processed embeddings
        min_cluster_size (int): Minimum number of documents required to form a cluster

    Returns:
        HDBSCAN: HDBSCAN model
    """
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
    """
    Adds cluster numbers to original Pandas DataFrame.

    Args:
        df (DataFrame): Pandas dataframe
        colname (str): Original column which you'd like to cluster on
        clusterer: HDBSCAN object from train_clusterer()

    Returns:
        Dataframe: Output DataFrame with topic numbers appended as a new column
    """

    df[f'{colname}_cluster'] = clusterer.labels_

    ## Casting the cluster number to a string, so the visuals get interpreted as discrete vals
    df[f'{colname}_cluster'] = df[f'{colname}_cluster'].astype(str)

    return df

def make_embedding_df(reduced_embeddings):
    """
    Converts reduced embeddings to a DataFrame, suitable for joining on the main workflow dataframe.

    Args:
        reduced_embeddings: UMAP-reduced embeddings

    Returns:
        DataFrame: Dataframe with embeddings broken out into cartesian components.
    """
    ## TODO: Smartly handle the new columns based on length of reduced_embeddings[0]
    df_emb = pd.DataFrame(reduced_embeddings).rename(columns={0:'x', 1:'y', 2:'z'})

    return df_emb
def add_reduced_cols(df, df_emb):
    """
    Glues/Merges X,Y, and Z features from UMAP reduction onto dataframe, to enable plotting.

    Args:
        df (DataFrame): Workflow DataFrame
        df_emb (DataFrame): DataFrame with cartesian coordinates, from make_embedding_df()

    Returns:
        df (DataFrame): Output dataframe, merging df and df_emb (right join)
    """


    ## Todo: Reduce the douple loop
    for zz in ['x', 'y', 'z']:
        if zz in df.columns:
            df = df.drop(columns=[zz])

    return pd.merge(df, df_emb, on=df.index.values).drop(columns='key_0')

def prep_for_viz(df, reduced_embeddings):
    """
    High-level utility function to create DF from reduced embeddings, and right-join that onto the workload DataFrame.

    Args:
        df (DataFrame): Workflow DataFrame
        reduced_embeddings: Embeddings arrays

    Returns:
        df (DataFrame): Output DataFrame with embeddings DataFrame joined onto it.
    """

    df_emb = make_embedding_df(reduced_embeddings)
    df = add_reduced_cols(df, df_emb)

    return df
def make_3d_scatter(df, cluster_col, unclustered=False):
    """
    Create a 3d scatterplot of reduced embeddings.

    Useful to examine the cluster quality of your topic model. Defaults to hiding unclustered documents. You can
    control this behavior by passing unclustered=True.

    Args:
        df (DataFrame): Workflow Pandas DataFrame (with XYZ columns from add_reduced_cols().
        cluster_col (str): Original column name you were clustering on.
        unclustered (bool): Whether or not to show unclustered data. Defaults to False (hiding unclustered docs).

    Returns:
        fig (plotly.express Figure): Plotly Express 3d Scatterplot of reduced embeddings.
    """

    if unclustered == False:
        df = df[df[f'{cluster_col}_cluster'] != "-1"]

    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color=f'{cluster_col}_cluster',
                        hover_data=['title'],
                        width=1000, height=1000
                        )
    fig.update_traces(marker={'size': 3})

    return fig


