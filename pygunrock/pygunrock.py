#!/usr/bin/env python

"""
    gunrock.py
"""

class BaseEnactor:
    def __init__(self):
        self.frontier = []
        self.stats = {'iteration' : 0}
    
    def Init(self, problem):
        self.problem = problem


class BaseIterationLoop():
    # !! TODO: Where does Gather get called?
    def __init__(self, enactor):
        self.enactor = enactor
    
    def Stop_Condition(self):
        return len(self.enactor.frontier) == 0
    
    def Core(self):
        enactor_stats = self.enactor.stats
        problem       = self.enactor.problem
        graph         = problem.graph
        
        new_frontier = []
        
        # Apply advance op
        for src in self.enactor.frontier:
            for dest in graph.neighbors(src):
                add_to_new_frontier = self._advance_op(src, dest, problem, enactor_stats)
                if add_to_new_frontier:
                    new_frontier.append(dest)
        
        # Apply filter op
        new_frontier = list(filter(lambda dest: self._filter_op(-1, dest, problem, enactor_stats), new_frontier))
        
        # Repeat
        self.enactor.frontier = new_frontier
    
    def run(self):
        while not self.Stop_Condition():
            self.Core()
            self.enactor.stats['iteration'] += 1
