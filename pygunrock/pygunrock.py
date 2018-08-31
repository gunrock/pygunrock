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
    
    def Enact(self, graph):
        while len(frontier):
            new_frontier = self.__iteration(
                graph=graph,
                frontier=self.frontier,
                problem=self.problem,
                advance_op=self._advance_op,
                filter_op=self._filter_op,
            )
            
            frontier = new_frontier
            
            self.stats['iteration'] += 1
        
        return frontier


class BaseIterationLoop():
    def run(self):
        
        enactor_stats = self.enactor.stats
        problem       = self.enactor.problem
        graph         = problem.graph
        
        while len(self.enactor.frontier):
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