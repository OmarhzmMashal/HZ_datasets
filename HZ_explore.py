import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
os.getcwd()
path = os.getcwd()
import uproot
import awkward as ak
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric.utils as py_utils
from scipy.special import rel_entr
import pandas as pd
import matplotlib
from sklearn.decomposition import PCA
from scipy.special import rel_entr
import seaborn as sns
from sklearn.decomposition import KernelPCA

# calculate the kl divergence
def kl_divergence(p, q):
    l=[]
    for i in range(len(p)):
        if p[i] == 0: p[i] = 1e-100
        if q[i] == 0: q[i] = 1e-100
        l.append(p[i] * np.log(p[i]/q[i]))
    return sum(l)

# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


path_to_data = "/HZdataset/"
path_to_save_plots = "plots"

features = ['pfcand_thetarel', 'pfcand_phirel','pfcand_dptdpt', 'pfcand_detadeta', 'pfcand_dphidphi', 'pfcand_dzdz', 'pfcand_dxydz', 'pfcand_dphidxy',
'pfcand_dxyc', 'pfcand_dxyctgtheta', 'pfcand_phic', 'pfcand_phictgtheta', 'pfcand_cdz', 'pfcand_cctgtheta', 'pfcand_mtof', 'pfcand_dndx',
'pfcand_charge', 'pfcand_isMu', 'pfcand_isEl', 'pfcand_isChargedHad', 'pfcand_isGamma','pfcand_isNeutralHad','pfcand_dxy', 'pfcand_dz']

coords = ['pfcand_thetarel', 'pfcand_phirel']

HQQ_df = HCC_df = ZQQ_df = ZCC_df = None
for i in tqdm(range(0,18)):
    try:
        file = uproot.open(os.getcwd()+f'{path_to_data}/hqq_train/_{i}.root')
        tree = file['deepntuplizer/tree']
        if HQQ_df is None:
            ind=0
            HQQ_df = ak.to_pandas(tree.arrays())[features]
        else:
            new = ak.to_pandas(tree.arrays())[features]
            new.index.set_levels(new.index.levels[0] + (ind+1) * (20000),0,inplace=True)
            HQQ_df = pd.concat((HQQ_df,new), axis=0, levels=0).sort_index(axis=0)
            ind+=1
    except:
        pass


for i in tqdm(range(0,18)):
    try:
        file = uproot.open(os.getcwd()+f'{path_to_data}/hcc_train/_{i}.root')
        tree = file['deepntuplizer/tree']
        if HCC_df is None:
            ind=0
            HCC_df = ak.to_pandas(tree.arrays())[features]
        else:
            new = ak.to_pandas(tree.arrays())[features]
            new.index.set_levels(new.index.levels[0] + (ind+1) * (20000),0,inplace=True)
            HCC_df = pd.concat((HCC_df,new), axis=0)
            ind+=1
    except:
        pass


for i in tqdm(range(0,18)):
    try:
        file = uproot.open(os.getcwd()+f'{path_to_data}/zqq_train/_{i}.root')
        tree = file['deepntuplizer/tree']
        if ZQQ_df is None:
            ind=0
            ZQQ_df = ak.to_pandas(tree.arrays())[features]
        else:
            new = ak.to_pandas(tree.arrays())[features]
            new.index.set_levels(new.index.levels[0] + (ind+1) * (20000),0,inplace=True)
            ZQQ_df = pd.concat((ZQQ_df,new), axis=0)
            ind+=1
    except:
        pass

for i in tqdm(range(0,18)):
    try:
        file = uproot.open(os.getcwd()+f'{path_to_data}/zcc_train/_{i}.root')
        tree = file['deepntuplizer/tree']
        if ZCC_df is None:
            ind=0
            ZCC_df = ak.to_pandas(tree.arrays())[features]
        else:
            new = ak.to_pandas(tree.arrays())[features]
            new.index.set_levels(new.index.levels[0] + (ind+1) * (20000),0,inplace=True)
            ZCC_df = pd.concat((ZCC_df,new), axis=0)
            ind+=1
    except:
        pass



class CreatGraph:
    def __init__(self, jet_df, k=7):
        self.prts_crds = np.array(jet_df[coords])
        self.prts_fts = np.array(jet_df[features])
        self.num_particles = len(self.prts_fts)
        self.k = k

    def get_k_nearest(self, point_source):
        dists = []
        dists = [np.sqrt(np.sum(np.square(point_source-point_target))) for point_target in self.prts_crds]
        nearest_k_idx = np.argsort(dists)[:self.k+1] # adding 1 becsuse the first one will be the point itself (self-loop)
        nearest_k_idx = nearest_k_idx[1:]
        return nearest_k_idx

    def get_edge_idx(self):
        edge_idx = []
        for source_idx, point_source in tqdm(enumerate(self.prts_crds)):
            nearest_k_idx = self.get_k_nearest(point_source)
            for target_idx in nearest_k_idx:
                pair = [source_idx, target_idx]
                edge_idx.append(pair)
        return np.array(edge_idx)

    def plot_graph(self, k=7):

        # construct edges knn
        edge_idx = self.get_edge_idx()

        # plot graph relative to 2D coords
        graph_locs_idx = {}
        for i in range(self.num_particles):
            graph_locs_idx[i] = (self.prts_crds[i,0], self.prts_crds[i,1])

        # plot graph
        graph = Data(x=torch.Tensor(self.prts_fts), edge_index=torch.Tensor(edge_idx).t().contiguous())
        g = py_utils.to_networkx(graph, to_undirected=True, remove_self_loops= True)
        #pz=[p[0]*1e2 for p in prts_fts]

        nx.draw(g, graph_locs_idx, node_color='red', node_size=10)
        plt.savefig("graph.png")
        plt.show()


# create a graph with nearest neighbours
jet_idx=30 # get a jet
jet_df = HCC_df.loc[(jet_idx,)]
graph = CreatGraph(jet_df, k=7)
graph.plot_graph()


# averaging the distribution for all jets
num_bins = 100
num_jets = 100
H_COLOR = "navy"
Z_COLOR = "red"
plt.rcParams.update({'font.size': 20})
features_prob_div=[]

for feature_name in features:

    plt.figure(figsize=(8,8))
    plt.axis("off")

    feature_dist_all = 0
    for j in tqdm(range(num_jets)):
        jet_feature = np.array(HQQ_df[feature_name].loc[(j,)])
        count, bins  = np.histogram(jet_feature, num_bins, density=True, normed=True)
        feature_dist_all += count
    HQQ_jets_avg = feature_dist_all/num_jets
    HQQ_range = bins[:len(bins)-1]

    feature_dist_all = 0
    for j in tqdm(range(num_jets)):
        jet_feature = np.array(HCC_df[feature_name].loc[(j,)])
        count, bins  = np.histogram(jet_feature, num_bins, density=True, normed=True)
        feature_dist_all += count
    HCC_jets_avg = feature_dist_all/num_jets
    HCC_range = bins[:len(bins)-1]

    feature_dist_all = 0
    for j in tqdm(range(num_jets)):
        jet_feature = np.array(ZQQ_df[feature_name].loc[(j,)])
        count, bins  = np.histogram(jet_feature, num_bins, density=True, normed=True)
        feature_dist_all += count
    ZQQ_jets_avg = feature_dist_all/num_jets
    ZQQ_range = bins[:len(bins)-1]

    feature_dist_all = 0
    for j in tqdm(range(num_jets)):
        jet_feature = np.array(ZCC_df[feature_name].loc[(j,)])
        count, bins  = np.histogram(jet_feature, num_bins, density=True, normed=True)
        feature_dist_all += count
    ZCC_jets_avg = feature_dist_all/num_jets
    ZCC_range = bins[:len(bins)-1]

    plt.plot(HQQ_range,HQQ_jets_avg, color = H_COLOR, label="H-QQ")
    plt.plot(HCC_range, HCC_jets_avg, color = H_COLOR, linestyle='dashed', label="H-CC")
    plt.plot(ZQQ_range, ZQQ_jets_avg, color = Z_COLOR, label="Z-QQ")
    plt.plot(ZCC_range, ZCC_jets_avg, color = Z_COLOR, linestyle='dashed', label="Z-CC")
    plt.legend(loc="upper right")

    plt.savefig(f"{path_to_save_plots}/{feature_name}.png")
    plt.show()

    # ensure equal number of bins for divergence summation
    HP = np.concatenate((HQQ_jets_avg,HCC_jets_avg), axis=0)
    ZP = np.concatenate((ZQQ_jets_avg,ZCC_jets_avg), axis=0)
    if HP.shape[0] > ZP.shape[0]:
        HP = HP[:ZP.shape[0]]
    else:
        ZP = ZP[:HP.shape[0]]
    prob_div = js_divergence(ZP, HP)
    features_prob_div.append(prob_div)


def plot_features_dists(features_prob_div, features):

    # clean names
    features_clean=[]
    for feature_name in features:
        features_clean.append(feature_name.partition("_")[2])
    features_clean=np.array(features_clean)
    features_prob_div = np.array(features_prob_div)

    num_features = 5 # choose tops
    idx = (-features_prob_div).argsort()[:num_features]
    topfeatures = features_clean[idx]
    topdivs = features_prob_div[idx]

    # normalize
    topdivs = abs((topdivs - topdivs.mean())/(topdivs.max()-topdivs.min()))
    plt.barh(np.flip(topfeatures), np.flip(topdivs), color="navy")

    plt.rc('ytick', labelsize=18)
    ax= plt.gca()
    #ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(f"{path_to_save_plots}/features_divs.png")
    plt.show()

plot_features_dists(features_prob_div, features)
