import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *

from time import perf_counter as pc


start_time = pc()

parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default='USAir', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from \
                    unknown links without overlap.')
parser.add_argument('--epoch', type=int, default=50, 
                    help='set maximum number of epoch')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-node2vec-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-spectral-embedding', action='store_true', default=False,
                    help='whether to use spectral node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    if 'group' in data:
        # load node attributes (here a.k.a. node classes)
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    train_pos, train_neg, test_pos, test_neg = sample_neg(net, args.test_ratio, max_train_num=args.max_train_num)
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, 
        train_pos=train_pos, 
        test_pos=test_pos, 
        max_train_num=args.max_train_num,
        all_unknown_as_negative=args.all_unknown_as_negative
    )


'''Train and apply classifier'''
A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

node_information = None
# node_information = np.sum(A.toarray(), axis=1, keepdims=True)  # easily overfit power
if args.use_node2vec_embedding:
    print(f"USING NODE2VEC EMBEDDING")
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    # embeddings = generate_spectral_embeddings(A)#, 128, True, train_neg)
    node_information = embeddings
elif args.use_spectral_embedding:
    print(f"USING SPECTRAL EMBEDDING")
    #embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    embeddings = generate_spectral_embeddings(A)#, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, node_information, args.no_parallel)
print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

# DGCNN configurations
cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
if args.no_cuda:
    cmd_args.mode = 'cpu'
else:
    cmd_args.mode = 'gpu'
cmd_args.num_epochs = args.epoch
cmd_args.learning_rate = 1e-4
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

train_idxes = list(range(len(train_graphs)))
best_loss = None
auc = [0.0, 0.001]
n_epoch = 0
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f roc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))

    classifier.eval()
    test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
    if not cmd_args.printAUC:
        test_loss[2] = 0.0
    print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f ap %.5f roc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3], test_loss[4]))

    auc[0] = auc[1] #previous
    auc[1] = test_loss[2] #current
    n_epoch = epoch
    if(auc[0]>auc[1]): #if previous is better, stop
        break

print(f"SEAL TIME: {round(pc() - start_time, 2)}")

if args.test_name == None:
    name = args.data_name
else:
    name = args.test_name

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\thop = ' + str(args.hop) + '\tepoch = ' + str(n_epoch) + '\t' + name + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(auc[0]) + '\thop = ' + str(args.hop) + '\tepoch = ' + str(n_epoch) + '\t' + name + '\n')

