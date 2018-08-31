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
        self.labels     = [None] * graph.number_of_nodes()
        self.preds      = [None] * graph.number_of_nodes()
        self.bc_values  = [0]    * graph.number_of_nodes()
        self.deltas     = [None] * graph.number_of_nodes()
        self.sigmas     = [None] * graph.number_of_nodes()
    
    def Reset(self, src):
        self.labels     = [None] * graph.number_of_nodes()
        self.preds      = [None] * graph.number_of_nodes()
        self.bc_values  = [0]    * graph.number_of_nodes()
        self.deltas     = [0]    * graph.number_of_nodes()
        self.sigmas     = [0]    * graph.number_of_nodes()
        
        self.labels[src] = 0
        self.preds[src]  = None
        self.sigmas      = 1
    
    def Extract(self):
        return list(zip(range(len(self.bc_values)), self.bc_values))


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


def cpu_reference(graph, parameters):
    return list(dict(graph.degree).items())


def validate_results(result1, result2):
    try:
        for r1, r2 in zip(result1, result2):
            assert r1 == r2
    except:
        raise Exception('validate_results failed!')


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
    
    problem.Reset(src)
    enactor.Reset(src)
    
    enactor.Enact(graph)
    
    gunrock_result = problem.Extract()
    
    reference_result = cpu_reference(graph, parameters)
    validate_results(reference_result, gunrock_result)
    
    print(gunrock_result)