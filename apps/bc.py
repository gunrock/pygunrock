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
# Problem

class Problem:
    def __init__(self, parameters):
        pass
    
    def Init(self, graph):
        self.graph = graph
        
        self.labels     = [None] * graph.number_of_nodes()
        self.preds      = [None] * graph.number_of_nodes()
        self.bc_values  = [0.0]  * graph.number_of_nodes()
        self.deltas     = [None] * graph.number_of_nodes()
        self.sigmas     = [None] * graph.number_of_nodes()
    
    def Reset(self, src):
        self.labels     = [None] * graph.number_of_nodes()
        self.preds      = [None] * graph.number_of_nodes()
        self.bc_values  = [0.0]  * graph.number_of_nodes()
        self.deltas     = [0]    * graph.number_of_nodes()
        self.sigmas     = [0]    * graph.number_of_nodes()
        
        self.labels[src] = 0
        self.preds[src]  = None
        self.sigmas[src] = 1
    
    def Extract(self):
        print(self.sigmas)
        assert self.sigmas == [1,9,1,2,3,3,1,1,4,3,1,1,1,1,1,1,1,1,3,2,2,1,1,2,1,2,2,2,1,3,2,3,3,1,1,6,1,5,1]
        # return list(zip(range(len(self.bc_values)), self.bc_values))


class ForwardIterationLoop(BaseIterationLoop):
    def _advance_op(self, src, dest, problem, enactor_stats):
        new_label = problem.labels[src] + 1
        old_label = problem.labels[dest]
        if old_label is None:
            problem.labels[dest] = new_label
        
        if (old_label != new_label) and (old_label is not None):
            return False
        
        problem.sigmas[dest] += problem.sigmas[src]
        
        return old_label is None
    
    def _filter_op(self, src, dest, problem, enactor_stats):
        return True


class BackwardIterationLoop(BaseIterationLoop):
    def _advance_op(self, src, dest, problem, enactor_stats):
        s_label = problem.labels[src]
        d_label = problem.labels[dest]
        
        if enactor_stats['iteration'] == 0:
            return d_label == s_label + 1
        else:
            if d_label == s_label + 1:
                if src == src_node:
                    return True
                
                from_sigma = problem.sigmas[src]
                to_sigma   = problem.sigmas[dest]
                to_delta   = problem.deltas[dest]
                result     = float(from_sigma) / to_sigma * (1 + to_delta)
                
                old_delta = problem.deltas[src]
                problem.deltas[src] += result
                
                old_bc_value = problem.bc_values[src]
                problem.bc_values[src] += result
                
                return True
            else:
                return False
    
    def _filter_op(self, src, dest, problem, enactor_stats):
        return problem.labels[dest] == 0
    
    def _gather(self):
        # !! TOOD: Something has to go here
        pass


class Enactor(BaseEnactor):
    def Reset(self, src):
        self.frontier = [src]
    
    def Enact(self):
        ForwardIterationLoop(self).run()
        # !! TODO gather not implemented!
        # BackwardIterationLoop(self).run()



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
    
    # reference_result = cpu_reference(graph, parameters)
    # validate_results(reference_result, gunrock_result)
    
    # print(gunrock_result)