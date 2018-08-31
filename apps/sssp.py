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

def cpu_reference(graph, parameters):
    sssp = nx.single_source_shortest_path_length(graph, parameters.src)
    sssp = dict(sssp).items()
    sssp = sorted(sssp, key=lambda x: x[0])
    return sssp

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
    
    def Reset(self, src):
        self.distances = [np.inf] * graph.number_of_nodes()
        self.labels    = [None] * graph.number_of_nodes()
        
        self.distances[src] = 0
    
    def Extract(self):
        return list(zip(range(len(self.distances)), self.distances))

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    def __init__(self, enactor):
        self.enactor = enactor
    
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
    def Reset(self, src):
        self.frontier.append(src)
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# --
# Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/chesapeake.edgelist')
    parser.add_argument('--src', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath, nodetype=int)
    
    src = parameters.src
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset(src)
    enactor.Reset(src)
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    
    reference_result = cpu_reference(graph, parameters)
    validate_results(reference_result, gunrock_result)
    
    print(gunrock_result)