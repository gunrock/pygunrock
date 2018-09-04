#!/usr/bin/env python

"""
    apps/hello.py
"""

import sys
import argparse
import networkx as nx
from pygunrock import BaseEnactor, BaseIterationLoop

import numpy as np

# --
# Helpers

def cpu_reference(graph, srcs):
    sssps = [nx.single_source_shortest_path_length(graph, src) for src in srcs]
    sssps = [dict(sssp).items() for sssp in sssps]
    sssps = [sorted(sssp, key=lambda x: x[0]) for sssp in sssps]
    
    res = []
    for x in zip(*sssps):
        min_dist = min([xx[1] for xx in x])
        res.append((x[0][0], min_dist))
    
    return res

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
        
        self.distances = [None] * graph.number_of_nodes()
        self.labels    = [None] * graph.number_of_nodes()
    
    def Reset(self, srcs):
        self.distances = [np.inf] * graph.number_of_nodes()
        self.labels    = [None] * graph.number_of_nodes()
        
        for src in srcs:
            self.distances[src] = 0
    
    def Extract(self):
        return list(zip(range(len(self.distances)), self.distances))

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    def _advance_op(self, src, dest, problem, enactor_stats):
        src_distance = problem.distances[src]
        edge_weight  = 1
        new_distance = src_distance + edge_weight
        
        old_distance = problem.distances[dest]
        problem.distances[dest] = min(problem.distances[dest], new_distance)
        
        return new_distance < old_distance
        
    def _filter_op(self, src, dest, problem, enactor_stats):
        if problem.labels[dest] == enactor_stats['iteration']:
            return False
        
        problem.labels[dest] = enactor_stats['iteration']
        return True

# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self, srcs):
        self.frontier = srcs
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# --
# Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/chesapeake.edgelist')
    parser.add_argument('--srcs', type=str, default='0,2')
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath, nodetype=int)
    
    srcs = [int(src) for src in parameters.srcs.split(',')]
    
    reference_result = cpu_reference(graph, srcs)
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset(srcs)
    enactor.Reset(srcs)
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    
    validate_results(reference_result, gunrock_result)
    
    print(gunrock_result)