import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt 


datafile_path = '../data/sample-youtube-dataset.txt'

df = pd.read_csv(datafile_path, delim_whitespace=True, header=None, names=['videoid', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'rating', 'comments', 'related1', 
    'related2', 'related3', 'related4', 'related5', 'related6', 'related7', 'related8', 'related9', 'related10', 'related11', 'related12', 'related13', 
    'related14', 'related15', 'related16', 'related17', 'related18', 'related19', 'related20'])
print('Dataset read')
print(df.head())
# df['item-rating'] = list(zip(df.itemid, df.rating))
videoids = df.videoid.unique()
features = []

G = nx.Graph()
G.add_nodes_from(videoids)
print('Nodes added')

edges = []

for index, row in df.iterrows():
    f = row[['age', 'length', 'views', 'rate', 'rating', 'comments']].astype('float64').to_list()
    features.append(f)
    for i in range(1, 21):
        node = row['related' + str(i)]
        if node in videoids:
            edges.append((row.videoid, node, 21 - i))

features = np.array(features)
print(features.shape)

G.add_weighted_edges_from(edges)
print('Edges added')

nx.draw(G)
plt.savefig('Item rating graph.png')

