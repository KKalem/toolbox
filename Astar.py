# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-12-21

import numpy as np
from . import geometry as geom

def Astar_search(s,
                 e,
                 cost_map,
                 use_diagonals = True,
                 heuristic_fn = geom.euclid_distance):
    """
    given a cost_map and two points on it, do A* search from point s to point e and return the path
    as an array of shape N,2.

    if use_diagonals==False, only the 4 on the plus shape are expanded, otherwise, diagonals are also used
    heuristic_fn should be a function that can take s and e as arguments and returns a single value.
    """
    s = tuple(s)
    e = tuple(e)

    closedset = set()
    openset = set()
    openset.add(s)

    came_from = {}

    # check for None when getting
    gScore = {}
    gScore[s] = 0

    fScore = {}
    fScore[s] = heuristic_fn(s,e)

    straight_cost = 1
    neighbors = [ [1,0,straight_cost], [-1,0,straight_cost], [0,1,straight_cost], [0,-1,straight_cost] ]

    if neighbors == 8:
        diagonal_cost = 1.42
        neighbors.extend( [[1,1,diagonal_cost], [-1,1,diagonal_cost], [1,-1,diagonal_cost], [-1,-1,diagonal_cost]] )

    while len(openset) > 0:
        # get the cheapest node and remove it
        fScore_list = list(fScore.values())
        fScore_node_list = list(fScore.keys())
        ix = np.argmin(fScore_list)
        current = fScore_node_list[ix]
        del fScore[current]

        if current == e:
            # found the goal
            # extract path from came_from
            path = []
            came_from_node = current
            while came_from_node is not None:
                path.append(came_from_node)
                came_from_node = came_from.get(came_from_node)

            return np.array(path)

        # looking at it
        openset.remove(current)
        closedset.add(current)

        # expand the node
        for dx,dy,cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor[0] not in range(0, cost_map.shape[0]) or neighbor[1] not in range(0, cost_map.shape[1]):
                # not in map
                continue

            if neighbor in closedset:
                # already looked at this
                continue

            tentative_gScore = gScore[current] + cost_map[neighbor[0], neighbor[1]] + cost

            if neighbor not in openset:
                openset.add(neighbor)
            elif tentative_gScore >= gScore[neighbor]:
                continue

            # we found the best ever path
            came_from[neighbor] = current
            gScore[neighbor] = tentative_gScore
            fScore[neighbor] = gScore[neighbor] + heuristic_fn(e, neighbor)

