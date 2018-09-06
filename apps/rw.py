#!/usr/bin/env python

"""
    apps/rw.py
"""

import sys
import argparse
import networkx as nx
from pygunrock import BaseEnactor, BaseIterationLoop

import numpy as np

# --
# Helpers

def validate_results(result, graph):
    # Check that this is a valid random walk
    for row in result:
        for src, dest in zip(row[:-1], row[1:]):
            assert (src, dest) in graph.edges, "(%d, %d) not in graph.edges" % (src, dest)


# --
# Data structure

class Problem:
    def __init__(self, parameters):
        self.walk_length = parameters.walk_length
    
    def Init(self, graph):
        self.graph = graph
        
        self.walks = [None] * graph.number_of_nodes() * self.walk_length
    
    def Reset(self):
        self.walks = [-1] * graph.number_of_nodes() * self.walk_length
    
    def Extract(self):
        return self.walks

# --
# Iteration loop

class IterationLoop(BaseIterationLoop):
    def Core(self):
        enactor_stats = self.enactor.stats
        frontier      = self.enactor.frontier
        problem       = self.enactor.problem
        graph         = problem.graph
        
        for i in range(len(frontier)):
            current_node = frontier[i]
            write_idx = (i * problem.walk_length) + enactor_stats['iteration']
            problem.walks[write_idx] = current_node
            
            if enactor_stats['iteration'] < problem.walk_length - 1:
                next_node    = np.random.choice(list(graph.neighbors(current_node)), 1)[0]
                frontier[i]  = next_node
    
    def Stop_Condition(self):
        enactor_stats = self.enactor.stats
        problem  = self.enactor.problem
        return enactor_stats['iteration'] == problem.walk_length

# --
# Enactor (wraps iteration loop)

class Enactor(BaseEnactor):
    def Reset(self):
        self.frontier = sorted([int(n) for n in self.problem.graph.nodes])
    
    def Enact(self):
        iteration_loop = IterationLoop(self)
        iteration_loop.run()

# --
# Run

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/chesapeake.edgelist')
    parser.add_argument('--walk-length', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath, nodetype=int)
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset()
    enactor.Reset()
    
    enactor.Enact()
    
    gunrock_result = problem.Extract()
    gunrock_result = np.array(gunrock_result).reshape(len(graph.nodes), -1)
    validate_results(gunrock_result, graph)
    print(gunrock_result)

