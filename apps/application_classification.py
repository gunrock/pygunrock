#!/usr/bin/env python

"""
    apps/pr_nibble.py
"""

import sys
import argparse
import pandas as pd
import networkx as nx
sys.path.append('/home/bjohnson/projects/davis/pygunrock')
from pygunrock import BaseEnactor, BaseIterationLoop
import numpy as np

from numba import jit
from scipy.spatial.distance import cdist
from reference.application_classification import application_classification

# --
# Helpers

def load_data():
    data_vertex    = pd.read_csv('./data/application_classification/georgiyData.Vertex.csv', skiprows=1, sep=' ', header=None)
    pattern_vertex = pd.read_csv('./data/application_classification/georgiyPattern.Vertex.csv', skiprows=1, sep=' ', header=None)
    data_edges     = pd.read_csv('./data/application_classification/georgiyData.Edges.csv', skiprows=1, sep=' ', header=None)
    pattern_edges  = pd.read_csv('./data/application_classification/georgiyPattern.Edges.csv', skiprows=1, sep=' ', header=None)
    
    data_vertex      = data_vertex.values[:,1:]
    data_edges_table = data_edges[list(range(2, data_edges.shape[1]))].values
    data_edges       = data_edges[[0, 1]].values
    
    pattern_vertex      = pattern_vertex.values[:,1:]
    pattern_edges_table = pattern_edges[list(range(2, pattern_edges.shape[1]))].values
    pattern_edges       = pattern_edges[[0, 1]].values
    
    return (
        data_vertex,
        data_edges,
        data_edges_table,
        pattern_vertex,
        pattern_edges,
        pattern_edges_table,
    )


def cpu_reference(*args, **kwargs):
    return application_classification(*args, **kwargs)

def validate_results(result1, result2):
    try:
        assert np.allclose(result1, result2)
    except:
        raise Exception('validate_results failed!')

# --
# Data structure

def _normprob(x):
    return np.log(np.exp(x) / np.exp(x).sum(axis=0, keepdims=True))

class Problem:
    def __init__(self, parameters):
        pass
        
    def Init(self, 
        graph,
        graph_vertex_feats,
        graph_edge_feats,
        pattern_edges,
        pattern_vertex_feats,
        pattern_edge_feats,
    ):
        
        self.num_dv = graph_vertex_feats.shape[0]
        self.num_de = graph_edge_feats.shape[0]
        
        self.num_pv = pattern_vertex_feats.shape[0]
        self.num_pe = pattern_edges.shape[0]
        
        self.vertex_feat_dim = graph_vertex_feats.shape[1]
        self.edge_feat_dim   = graph_edge_feats.shape[1]
        
        self.cv = np.zeros((self.num_dv, self.num_pv))
        self.mu = np.zeros((self.num_dv, self.num_pv))
        self.cv_max = np.zeros(self.num_pv) - np.inf
        self.cv_sum = np.zeros(self.num_pv)
        self.mu_max = np.zeros(self.num_pv) - np.inf
        self.mu_sum = np.zeros(self.num_pv)
        
        self.vf = np.zeros((self.num_dv, self.num_pe))
        self.vr = np.zeros((self.num_dv, self.num_pe))
        
        self.ce = np.zeros((self.num_de, self.num_pe))
        self.xe = np.zeros((self.num_de, self.num_pe))
        self.ce_max = np.zeros(self.num_pe) - np.inf
        self.ce_sum = np.zeros(self.num_pe)
        self.xe_max = np.zeros(self.num_pe) - np.inf
        self.xe_sum = np.zeros(self.num_pe)
        
        self.re = np.zeros((self.num_de, self.num_pe))
        self.fe = np.zeros((self.num_de, self.num_pe))
        
        
        # Revisit this later
        self.cnull = np.zeros(self.num_pe)
        
        self.graph                = graph
        self.graph_vertex_feats   = graph_vertex_feats
        self.graph_edge_feats     = graph_edge_feats
        self.pattern_edges        = pattern_edges
        self.pattern_vertex_feats = pattern_vertex_feats
        self.pattern_edge_feats   = pattern_edge_feats
    
    def Reset(self):
        self.cv = np.zeros((self.num_dv, self.num_pv))
        self.mu = np.zeros((self.num_dv, self.num_pv))
        self.vf = np.zeros((self.num_dv, self.num_pe))
        self.vr = np.zeros((self.num_dv, self.num_pe))
        self.ce = np.zeros((self.num_de, self.num_pe))
        self.re = np.zeros((self.num_de, self.num_pe))
        self.fe = np.zeros((self.num_de, self.num_pe))
    
    # def Extract(self):
    #     return np.abs(self.q * self.d_sqrt)

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    
    def _vertex_distance(self, enactor, i):
        prob = enactor.problem
        g_feat = prob.graph_vertex_feats[i]
        for j in range(prob.num_pv):
            p_feat = prob.pattern_vertex_feats[j]
            tmp = np.sqrt(((g_feat - p_feat) ** 2).sum())
            prob.cv[i, j] = tmp
            prob.mu[i, j] = -tmp
            
            prob.cv_max[j] = max(prob.cv_max[j], tmp)
            prob.mu_max[j] = max(prob.mu_max[j], -tmp)
    
    def _vertex_sub_max(self, enactor, j):
        prob = enactor.problem
        for i in range(prob.num_dv):
            prob.cv_sum[j] += np.exp(prob.cv[i, j] - prob.cv_max[j])
            prob.mu_sum[j] += np.exp(prob.mu[i, j] - prob.mu_max[j])
    
    def _vertex_col_norm(self, enactor, i):
        prob = enactor.problem
        for j in range(prob.num_pv):
            prob.cv[i, j] = np.log(np.exp(prob.cv[i, j] - prob.cv_max[j]) / prob.cv_sum[j])
            prob.mu[i, j] = np.log(np.exp(prob.mu[i, j] - prob.mu_max[j]) / prob.mu_sum[j])
    
    # --
    
    def _edge_distance(self, enactor, i):
        prob = enactor.problem
        g_feat = prob.graph_edge_feats[i]
        for j in range(prob.num_pe):
            p_feat = prob.pattern_edge_feats[j]
            tmp = np.sqrt(((g_feat - p_feat) ** 2).sum())
            prob.ce[i, j] = tmp
            prob.xe[i, j] = -tmp
            
            prob.ce_max[j] = max(prob.ce_max[j], tmp)
            prob.xe_max[j] = max(prob.xe_max[j], -tmp)
    
    def _edge_sub_max(self, enactor, j):
        prob = enactor.problem
        for i in range(prob.num_de):
            prob.ce_sum[j] += np.exp(prob.ce[i, j] - prob.ce_max[j])
            prob.xe_sum[j] += np.exp(prob.xe[i, j] - prob.xe_max[j])
    
    def _edge_col_norm(self, enactor, i):
        prob = enactor.problem
        for j in range(prob.num_pe):
            prob.ce[i, j] = np.log(np.exp(prob.ce[i, j] - prob.ce_max[j]) / prob.ce_sum[j])
            tmp = np.log(np.exp(prob.xe[i, j] - prob.xe_max[j]) / prob.xe_sum[j])
            prob.re[i, j] = tmp
            prob.fe[i, j] = tmp
    
    def Core(self):
        frontier = self.enactor.frontier
        prob     = self.enactor.problem
        
        iteration = self.enactor.stats['iteration']
        if iteration == 0:
            # initialize cv + mu
            for i in range(len(frontier)):
                self._vertex_distance(enactor, frontier[i])
            
            for pv_idx in range(prob.num_pv):
                self._vertex_sub_max(enactor, pv_idx)
            
            for i in range(len(frontier)):
                self._vertex_col_norm(enactor, frontier[i])
            
            for i, (src, dst) in enumerate(prob.pattern_edges):
                prob.vr[:,i] = prob.mu[:,src]
                prob.vf[:,i] = prob.mu[:,dst]
            
            # initialize ce + re + fe (can happen in parallel to above)
            for de_idx in range(prob.num_de):
                self._edge_distance(enactor, de_idx)
            
            for pe_idx in range(prob.num_pe):
                self._edge_sub_max(enactor, pe_idx)
            
            for de_idx in range(prob.num_de):
                self._edge_col_norm(enactor, de_idx)
            
            raise Exception
    
    def _advance_op(self, src, dest, problem, enactor_stats):
        pass
    
    def _filter_op(self, src, dest, problem, enactor_stats):
        return True
    
    def Stop_Condition(self):
        iteration = self.enactor.stats['iteration']
        return iteration == 1
        # iteration = self.enactor.stats['iteration']
        # prob      = self.enactor.problem
        # return iteration == 

# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self):
        num_dv = len(self.problem.graph.nodes)
        self.frontier = list(range(num_dv))
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# # --
# # Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == '__main__':
    parameters = parse_parameters()
    
    (graph_vertex_feats, graph_edges, graph_edge_feats, 
        pattern_vertex_feats, pattern_edges, pattern_edge_feats) = load_data()
    
    reference_result = cpu_reference(graph_vertex_feats, graph_edges, graph_edge_feats, 
        pattern_vertex_feats, pattern_edges, pattern_edge_feats, num_iters=0)
    # print(reference_result)
    
    graph = nx.from_edgelist(graph_edges)
    
    problem = Problem(parameters)
    problem.Init(
        graph=graph,
        graph_vertex_feats=graph_vertex_feats,
        graph_edge_feats=graph_edge_feats, 
        pattern_edges=pattern_edges,
        pattern_vertex_feats=pattern_vertex_feats,
        pattern_edge_feats=pattern_edge_feats
    )
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset()
    enactor.Reset()
    
    enactor.Enact()
    
#     # gunrock_result = problem.Extract()
#     # validate_results(reference_result, gunrock_result)
#     # print('passed test!')