#!/usr/bin/env python

"""
    gunrock.py
"""

class BaseEnactor:
    def __init__(self):
        self.frontier = []
    
    def Init(self, problem):
        self.problem = problem
    
    def Enact(self, graph):
        while len(self.frontier):
            new_frontier = self.__iteration(
                graph=graph,
                frontier=self.frontier,
                problem=self.problem,
                advance_op=self._advance_op,
                filter_op=self._filter_op,
            )
            
            self.frontier = new_frontier
    
    def __iteration(self, graph, frontier, problem, advance_op, filter_op):
        new_frontier = []
        
        # Apply advance op
        for src in frontier:
            for dest in graph.neighbors(src):
                add_to_new_frontier = advance_op(src, dest, problem)
                if add_to_new_frontier:
                    new_frontier.append(dest)
        
        # Apply filter op
        new_frontier = list(filter(lambda dest: filter_op(-1, dest, problem), new_frontier))
        
        return new_frontier
