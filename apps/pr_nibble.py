#!/usr/bin/env python

"""
    apps/pr_nibble.py
    
    !! Only works for undirected graphs
        !! Because it's not obvious how to manually add a node to the frontier,
            I'm doing a little hack where I initialize the algorithm using
            the `ref_node` and one of it's neighbors, and then ignore the neighbor
            on the first iteration.  In an undirected graph, this ensures that 
            every node that's been touched remains in the frontier.
            
            In a directed graph, I think we could add a dummy node connected w/
            a bidirectional edge to the source and initialize the algorithm 
            w/ those two nodes.  _This needs to be tested though._
    
    !! Only running w/ single `ref_node` ATM, but could easily be changed
    to take multiple nodes.  Just need to put a bunch of initialization stuff inside
    loops
    
    !! Probably does not support self-loops?
"""

import sys
import argparse
import networkx as nx
sys.path.append('/home/bjohnson/projects/davis/pygunrock')
from pygunrock import BaseEnactor, BaseIterationLoop

import numpy as np

try:
    from localgraphclustering import pageRank_nibble, graph_class_local
except:
    raise Exception('!! must install localgraphclustering')

# --
# Helpers

def cpu_reference(parameters):
    graph = graph_class_local.GraphLocal(parameters.inpath, 'edgelist', '\t')
    pr_nb = pageRank_nibble.PageRank_nibble()
    vector = pr_nb.produce([graph], [parameters.src], vol=parameters.vol, iterations=parameters.max_iter)[0]
    return vector

def validate_results(result1, result2):
    try:
        assert np.allclose(result1, result2)
    except:
        raise Exception('validate_results failed!')

# --
# Data structure

class Problem:
    def __init__(self, parameters):
        self.max_iter = parameters.max_iter
        self.phi      = parameters.phi
        self.eps      = parameters.eps
        self.vol      = parameters.vol if parameters.vol != 0 else 1
        
    def Init(self, graph):
        self.graph = graph
        
        # Init data structures
        num_nodes = graph.number_of_nodes()
        self.grad    = [0 for _ in range(num_nodes)]
        self.q       = [0 for _ in range(num_nodes)]
        self.y       = [0 for _ in range(num_nodes)]
        self.z       = [0 for _ in range(num_nodes)]
        self.touched = [False for _ in range(num_nodes)]
        
        self.d       = [len(list(graph.neighbors(n))) for n in sorted(graph.nodes)]
        self.d_sqrt  = np.sqrt(self.d)
        self.dn_sqrt = 1.0 / np.sqrt(self.d)
        
        # Init parameters based on defaults
        num_edges = len(graph.edges)
        log_num_edges = np.log2(num_edges)
        
        self.alpha = (self.phi ** 2) / (225 * np.log(100 * np.sqrt(num_edges)))
        
        self.rho = 1 / (2 ** min(1 + np.log2(self.vol), log_num_edges))
        self.rho *= (1 / (48 * log_num_edges))
    
    def Reset(self, ref_node, ref_neib):
        # Clear data structures
        self.grad    = [0] * len(self.grad)
        self.q       = [0] * len(self.q)
        self.y       = [0] * len(self.y)
        self.z       = [0] * len(self.z)
        self.touched = [False] * len(self.touched)
        
        self.num_ref_nodes  = 1
        self.grad[ref_node] = - self.alpha * self.dn_sqrt[ref_node] / self.num_ref_nodes
        
        thresh = self.rho * self.alpha * self.d_sqrt[ref_node]
        if (-self.grad[ref_node] < thresh).all():
            raise Exception('!! invalid initialization')
        
        self.ref_node = ref_node
        self.ref_neib = ref_neib
    
    def Extract(self):
        return np.abs(self.q * self.d_sqrt)

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    
    def _node_lambda(self, enactor, idx):
        iteration = self.enactor.stats['iteration']
        frontier  = self.enactor.frontier
        prob      = self.enactor.problem
        
        # skip the neighbor on the first iteration
        if (iteration == 0) and (idx == prob.ref_neib):
            return
        
        # this is at end in original implementation, but works here after the first iteration
        if (idx == prob.ref_node) and (iteration > 0):
            prob.grad[idx] -= prob.alpha / prob.num_ref_nodes * prob.dn_sqrt[prob.ref_node]
        
        prob.z[idx] = prob.y[idx] - prob.grad[idx]
        if prob.z[idx] == 0:
            return
        else:
            q_old  = prob.q[idx]
            thresh = prob.rho * prob.alpha * prob.d_sqrt[idx]
            if prob.z[idx] >= thresh:
                prob.q[idx] = prob.z[idx] - thresh
            elif prob.z[idx] <=  -thresh:
                prob.q[idx] = prob.z[idx] + thresh
            else:
                prob.q[idx] = 0
        
        if iteration == 0:
            prob.y[idx] = prob.q[idx]
        else:
            beta = (1 - np.sqrt(prob.alpha)) / (1 + np.sqrt(prob.alpha))
            prob.y[idx] = prob.q[idx] + beta * (prob.q[idx] - q_old)
        
        prob.touched[idx] = False
        prob.grad[idx] = prob.y[idx] * (1 + prob.alpha) / 2
    
    def Core(self):
        frontier = self.enactor.frontier
        
        # Node map
        for i in range(len(frontier)):
            self._node_lambda(enactor, frontier[i])
        
        # Advance/filter
        super().Core()
    
    def _advance_op(self, src, dest, problem, enactor_stats):
        already_touched = problem.touched[dest]
        
        val = problem.dn_sqrt[src] * problem.y[src] * problem.dn_sqrt[dest]
        problem.grad[dest] -= (val * (1 - problem.alpha) / 2)
        if (problem.grad[dest] != 0) and (not already_touched):
            problem.touched[dest] = True
            return True
        else:
            return False
    
    def _filter_op(self, src, dest, problem, enactor_stats):
        return True
    
    def Stop_Condition(self):
        iteration = self.enactor.stats['iteration']
        prob      = self.enactor.problem
        
        # too many iterations
        break_iter = iteration == prob.max_iter
        
        # too small gradient
        prob_grad = np.array(prob.grad).copy()
        if iteration > 0:
            prob_grad[prob.ref_node] -= prob.alpha / prob.num_ref_nodes * prob.dn_sqrt[prob.ref_node]
        
        max_grad   = np.abs(prob_grad * np.array(prob.dn_sqrt)).max()
        break_grad = max_grad <= prob.rho * prob.alpha * (1 + prob.eps)
        
        return break_iter or break_grad


# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self, ref_node, ref_neib):
        self.frontier = [ref_node, ref_neib]
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# --
# Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/jhu.edgelist.mapped')
    parser.add_argument('--src', type=int, default=0)
    
    parser.add_argument('--vol', type=int, default=40)
    parser.add_argument('--phi', type=float, default=0.5)
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--eps', type=float, default=1e-2)
    
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath, nodetype=int)
    
    src = parameters.src
    src_neib = list(graph.neighbors(src))[0]
    
    reference_result = cpu_reference(parameters)
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset(src, src_neib)
    enactor.Reset(src, src_neib)
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    validate_results(reference_result, gunrock_result)
    print('passed test!')