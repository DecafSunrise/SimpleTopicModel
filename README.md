# SimpleTopicModel
Easily identifying themes in text

![image](https://github.com/DecafSunrise/SimpleTopicModel/assets/36832027/3555d505-392a-4729-bdc6-9256079a376d)

## What is this?
This is a package that wraps up common theme identification (Topic Modeling) techniques in Python. SimpleTopicModel is currently under development, and subject to change.

## How do I get it?
Currently, you can git clone this repo and import it locally. Be sure to run `pip install -r requirements.txt` in the repo folder, to ensure you've got the relevant requirements.

I'm working on setting up a pypi release, slated for the near future.

## How do I use it?
![image](https://github.com/DecafSunrise/SimpleTopicModel/assets/36832027/b32ef2d3-c9fd-4304-864a-3d873c5bbe7e)
Use couldn't be easier. Most topic modeling techniques follow the same paradigm:
1. **Convert your text to numbers (embeddings)**: The excellent `Sentence-Transformers` package does this for us, using Microsoft's Mini-LM model.
2. **Reduce Dimensionality**: This package uses `UMAP`, but you could substitute TSNE or PCA if you wanted to.
3. **Cluster**: We're using `HDBSCAN` to build hierarchial clusters (which we'd like to traverse in a later release), but you could also use a KNN, GMM, etc.
4. **Visualize (Optional)**: This displays the reduced dimension embeddings in 3d (or 2d) space, so you can get a feel for how "tight" the clusters are.

## What's next?
- Clean up/professionalize this repo & releases
- Add automated cluster naming techniques (cTF-IDF, LLM-assisted naming, etc)
- Make a sweet logo & eyecatching graphics
- Fix the docs page

## Acknowledgements:
This builds on previous work including [Gensim (LDA)](https://radimrehurek.com/gensim/), [BERTopic](https://maartengr.github.io/BERTopic/index.html), [Top2Vec](https://github.com/ddangelov/Top2Vec), and [pyLDAvis](https://pypi.org/project/pyLDAvis/). They're all excellent, more mature alternatives to SimpleTopicModel, and I'd encourage you to go check them out!
