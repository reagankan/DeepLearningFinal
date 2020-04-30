from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt 

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
tf.set_random_seed(0)
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Calculate ROC AUC
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        pos.append(adj_orig[e[0], e[1]]) # actual value (1)

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]])) # predicted score for given edge
        neg.append(adj_orig[e[0], e[1]]) # actual value (0)

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

start = time.time()

datafile_path = '../data/youtube-dataset.txt'

df = pd.read_csv(datafile_path, delim_whitespace=True, header=None, names=['videoid', 'uploader', 'age', 'category', 'length', 'views', 'rate', 'rating', 'comments', 'related1', 
    'related2', 'related3', 'related4', 'related5', 'related6', 'related7', 'related8', 'related9', 'related10', 'related11', 'related12', 'related13', 
    'related14', 'related15', 'related16', 'related17', 'related18', 'related19', 'related20'])
print('Dataset read - ' + str(time.time() - start))
# df = df.iloc[0:500]
df = df.fillna(df.mean())
df['row_number'] = np.arange(len(df))
print(df.head())

# df['item-rating'] = list(zip(df.itemid, df.rating))
videoids = df.videoid.unique()
features = []
videoid_to_row = {v:i for i, v in enumerate(videoids)}
d = df.set_index('videoid').to_dict()
videoid_with_views = d['views']
videoid_with_rating = d['rating']
# print(videoid_to_row)
# print(videoid_with_views)
# print(videoid_with_rating)

G = nx.read_gpickle('../data/graph-20-related.g')

# G = nx.Graph()
# G.add_nodes_from(videoids)
# print('Nodes added - ' + str(time.time() - start))

# edges = []
# num_related = 20
# cat_count = {}
# cat_out_edges = {}

# for index, row in df.iterrows():
#     f = row[['age', 'length', 'views', 'rate', 'rating', 'comments']].astype('float64').to_list()
#     features.append(f)

#     if row.category in cat_count.keys():
#         cat_count[row.category] += 1
#     else:
#         cat_count[row.category] = 1

#     if row.category not in cat_out_edges:
#         cat_out_edges[row.category] = 0

#     for i in range(1, 1 + num_related):
#         node = row['related' + str(i)]
#         if node in videoids:
#             edges.append((row.videoid, node, 1))
#             row_index = videoid_to_row[node]
#             node_row = df.iloc[row_index]
#             if row.category != node_row.category:
#                 cat_out_edges[row.category] += 1

# print('Rows processed - ' + str(time.time() - start))
# print(sorted(cat_count.items(), reverse=True, key=lambda x: x[1]))
# print(sorted(cat_out_edges.items(), reverse=True, key=lambda x: x[1]))

# G.add_weighted_edges_from(edges)
# print('Edges added - ' + str(time.time() - start))

# nx.write_gpickle(G, '../data/graph-20-related-unweighted.g')

# features = np.array(features)
# features = (features - features.min(0)) / features.ptp(0)

# np.save('../data/features.npy', features)

features = np.load('../data/features.npy')
num_nodes = 1000

node_with_rating = sorted(list(videoid_with_rating.items()), key=lambda x: x[1], reverse=True)[:num_nodes]
nodes = [x[0] for x in node_with_rating]
node_rows = [videoid_to_row[node] for node in nodes]
features = features[node_rows, :]

adj = nx.adjacency_matrix(G, nodes)
print('Generated adjacency_matrix - ' + str(time.time() - start))

sp.save_npz('../data/adj-1000-rating.npz', adj)
# adj = sp.load_npz('../data/adj-deg-1000.npz')

print(features.shape)
print(adj.shape)

x = sp.lil_matrix(features)
features_tuple = sparse_to_tuple(x)
features_shape = features_tuple[2]

# Get graph attributes (to feed into model)
num_nodes = adj.shape[0] # number of nodes in adjacency matrix
num_features = features_shape[1] # number of features (columsn of features matrix)
features_nonzero = features_tuple[1].shape[0] # number of non-zero entries in features matrix (or length of values list)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

print('Some preprocessing done - ' + str(time.time() - start))

np.random.seed(0) # IMPORTANT: guarantees consistent train/test splits
adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
    test_edges, test_edges_false = mask_test_edges(adj, test_frac=.3, val_frac=.1)

print('Splitting done - ' + str(time.time() - start))

# Normalize adjacency matrix
adj_norm = preprocess_graph(adj_train)

print('Preprocessed graph - ' + str(time.time() - start))

# Add in diagonals
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Inspect train/test split
print("Total nodes:", adj.shape[0])
print("Total edges:", int(adj.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
print("Training edges (positive):", len(train_edges))
print("Training edges (negative):", len(train_edges_false))
print("Validation edges (positive):", len(val_edges))
print("Validation edges (negative):", len(val_edges_false))
print("Test edges (positive):", len(test_edges))
print("Test edges (negative):", len(test_edges_false))

# Define hyperparameters
LEARNING_RATE = 0.005
EPOCHS = 300
HIDDEN1_DIM = 32
HIDDEN2_DIM = 16
DROPOUT = 0.1

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

# How much to weigh positive examples (true edges) in cost print_function
  # Want to weigh less-frequent classes higher, so as to prevent model output bias
  # pos_weight = (num. negative samples / (num. positive samples)
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

# normalize (scale) average weighted cost
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Create VAE model
model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,
                   HIDDEN1_DIM, HIDDEN2_DIM)

opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           learning_rate=LEARNING_RATE)

cost_val = []
acc_val = []
val_roc_score = []

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Initialized variables - ' + str(time.time() - start))

# Train model
for epoch in range(EPOCHS):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
    feed_dict.update({placeholders['dropout']: DROPOUT})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    # Evaluate predictions
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    # Print results for this epoch
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - start))

print("Optimization Finished!")

# Print final results
roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
