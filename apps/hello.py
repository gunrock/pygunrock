#!/usr/bin/env python

"""
    apps/hello.py
"""

import sys
import argparse
import networkx as nx
from pygunrock import BaseEnactor

class Problem:
    def __init__(self, parameters):
        pass
    
    def Init(self, graph):
        self.degrees = [0] * graph.number_of_nodes()
        self.visited = [0] * graph.number_of_nodes()
    
    def Reset(self):
        self.degrees = [0] * graph.number_of_nodes()
        self.visited = [0] * graph.number_of_nodes()
    
    def Extract(self):
        return list(zip(range(len(self.degrees)), self.degrees))


class Enactor(BaseEnactor):
    def Reset(self, src):
        self.frontier.append(src)
    
    def _advance_op(self, src, dest, problem):
        problem.visited[src] = 1
        
        dest_visited  = problem.visited[dest]
        problem.visited[dest] = 1
        
        problem.degrees[src] += 1
        
        return dest_visited == 0
        
    def _filter_op(self, src, dest, problem):
        return True


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/chesapeake.edgelist')
    parser.add_argument('--src', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    parameters = parse_parameters()
    
    graph = nx.read_edgelist(parameters.inpath)
    graph = nx.convert_node_labels_to_integers(graph)
    
    src = parameters.src
    
    problem = Problem(parameters)
    problem.Init(graph)
    
    enactor = Enactor()
    enactor.Init(problem)
    
    problem.Reset()
    enactor.Reset(src)
    
    enactor.Enact(graph)
    
    result = problem.Extract()
    print(result)
    
    reference = dict(graph.degree).items()
    for a, b in zip(reference, result):
        assert a == b