#!/usr/bin/env python

"""
    apps/hello.py
"""

import sys
import argparse
import networkx as nx
from pygunrock import BaseEnactor, BaseIterationLoop

# --
# Helpers

def cpu_reference(graph, parameters):
    return list(dict(graph.degree).items())

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
        
        self.degrees = [0] * graph.number_of_nodes()
        self.visited = [0] * graph.number_of_nodes()
    
    def Reset(self):
        self.degrees = [0] * graph.number_of_nodes()
        self.visited = [0] * graph.number_of_nodes()
    
    def Extract(self):
        return list(zip(range(len(self.degrees)), self.degrees))

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    def __init__(self, enactor):
        self.enactor = enactor
    
    def _advance_op(self, src, dest, problem, enactor_stats):
        problem.visited[src] = 1
        
        dest_visited  = problem.visited[dest]
        problem.visited[dest] = 1
        
        problem.degrees[src] += 1
        
        return dest_visited == 0
        
    def _filter_op(self, src, dest, problem, enactor_stats):
        return True

# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self, src):
        self.frontier = [src]
    
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
    
    problem.Reset()
    enactor.Reset(src)
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    
    reference_result = cpu_reference(graph, parameters)
    validate_results(reference_result, gunrock_result)
    
    print(gunrock_result)