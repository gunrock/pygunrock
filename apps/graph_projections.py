#!/usr/bin/env python

"""
    apps/hello.py
"""

import sys
import argparse
import networkx as nx
from pygunrock import BaseEnactor, BaseIterationLoop

from collections import Counter

# --
# Helpers

def cpu_reference(graph, parameters):
    out = Counter()
    for node in graph.nodes():
        for neighbor_1 in graph.neighbors(node):
            for neighbor_2 in graph.neighbors(node):
                if neighbor_1 != neighbor_2:
                    out[(neighbor_1, neighbor_2)] += 1
    
    return sorted(list(out.items()))

def validate_results(result1, result2):
    try:
        for r1, r2 in zip(result1, result2):
            assert r1 == r2
    except:
        raise Exception('validate_results failed!')

# --
# Data structure

class Problem:
    def __init__(self, parameters):
        pass
    
    def Init(self, graph):
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.max_edges = self.num_nodes ** 2
        self.projection_edges = [0] * self.max_edges
    
    def Reset(self):
        self.projection_edges = [0] * (graph.number_of_nodes() ** 2)
    
    def Extract(self):
        out = []
        for idx, val in enumerate(self.projection_edges):
            if val != 0:
                row = idx // self.num_nodes
                col = idx % self.num_nodes
                out.append(((row, col), val))
        
        return out

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    def __init__(self, enactor):
        self.enactor = enactor
    
    def _advance_op(self, src, dest, problem, enactor_stats):
        for neib in problem.graph.neighbors(src):
            if dest != neib:
                problem.projection_edges[dest * problem.num_nodes + neib] += 1
        
        return False
        
    def _filter_op(self, src, dest, problem, enactor_stats):
        return True

# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self):
        self.frontier = list(range(self.problem.num_nodes))
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# --
# Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/graph_projections_sample.tsv')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath, nodetype=int, create_using=nx.DiGraph())
    reference_result = cpu_reference(graph, parameters)
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset()
    enactor.Reset()
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    
    validate_results(reference_result, gunrock_result)
    
    for edge in gunrock_result:
        print(edge)
