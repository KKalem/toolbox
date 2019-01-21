# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-12-21

import numpy as np

try:
    from . import geometry as geom
except SystemError:
    import geometry as geom


def closest_euclidian(start, goals):
    goal_shape = np.array(goals).shape
    if goal_shape == (2,):
        # single goal put it in a list to make the rest general
        goals = [goals]

    costs = [geom.euclid_distance(start, goal) for goal in goals]
    return min(costs)


def closests_manhattan(start, goals):
    def manhattan(start, goal):
        return abs(start[0]-goal[0])+abs(start[1]-goal[1])

    goal_shape = np.array(goals).shape
    if goal_shape == (2,):
        # single goal put it in a list to make the rest general
        goals = [goals]

    costs = [manhattan(start, goal) for goal in goals]
    return min(costs)

def Astar_search(s,
                 e,
                 cost_map,
                 use_diagonals = True,
                 heuristic_fn = closests_manhattan,
                 forbidden_map = None,
                 free_map = None):
    """
    given a cost_map and two points on it, do A* search from point s to point e and return the path
    as an array of shape N,2.

    if use_diagonals==False, only the 4 on the plus shape are expanded, otherwise, diagonals are also used
    heuristic_fn should be a function that can take s and e as arguments and returns a single value.

    s and e can be lists, at which point the heuristic_fn should be able to handle the case where the second
    argument could be a list.

    forbidden_map is an array the same shape as cost_map, the search will not use the cells>0 in the forbidden_map.
    free_map is the opposite, cells marked >0 will not even have traversal cost with them.
    """
    if forbidden_map is not None:
        assert forbidden_map.shape == cost_map.shape, "fobidden map and cost map should be the same shape!"

    try:
        ret_shape = np.array(heuristic_fn([3,4], [[1,2],[2,3]])).shape
        assert ret_shape == (), "heuristic_fn should return a single value even when given a list as args!"
    except:
        assert False, "heuristic_fn could not accept [3,4] as first arg and [[1,2],[2,3]] as second arg!"


    # check the shape of the inputs, they might be lists of points instead of single points
    s_shape = np.array(s).shape
    e_shape = np.array(e).shape

    # s and e are single points, make them into lists instead
    if s_shape == (2,):
        starts = [tuple(s)]
    else:
        # s is already some kind of a list, check the dimensions
        assert s_shape[1] == 2, "s can not have a shape larger than (N,2)"
        starts = [tuple(pt) for pt in s]


    if e_shape == (2,):
        ends = [tuple(e)]
    else:
        # e is already some kind of a list, check the dimensions
        assert e_shape[1] == 2, "e can not have a shape larger than (N,2)"
        ends = []
        for pt in e:
            # check if an ending point is actually forbidden
            if forbidden_map is not None and forbidden_map[pt[0], pt[1]] > 0:
                # its forbidden, so not a valid end point
                continue
            ends.append(tuple(pt))



    closedset = set()
    openset = set()
    # check for None when getting
    # real cost of a point, this doesnt include the heuristic estimate
    real_costs = {}
    heuristic_costs = {}
    came_from = {}

    for s in starts:
        s = tuple(s)
        openset.add(s)
        real_costs[s] = 0
        heuristic_costs[s] = heuristic_fn(s, ends)


    straight_cost = 1
    neighbors = [ [1,0,straight_cost], [-1,0,straight_cost], [0,1,straight_cost], [0,-1,straight_cost] ]

    if neighbors == 8:
        diagonal_cost = 1.42
        neighbors.extend( [[1,1,diagonal_cost], [-1,1,diagonal_cost], [1,-1,diagonal_cost], [-1,-1,diagonal_cost]] )

    while len(openset) > 0:
        # get the cheapest node and remove it
        heuristic_cost = list(heuristic_costs.values())
        heuristic_cost_nodes = list(heuristic_costs.keys())
        ix = np.argmin(heuristic_cost)
        current = heuristic_cost_nodes[ix]
        del heuristic_costs[current]

        if current in ends:
            # found the goal
            # extract path from came_from
            path = []
            total_cost = 0
            came_from_node = current
            while came_from_node is not None:
                path.append(came_from_node)
                total_cost += cost_map[came_from_node[0], came_from_node[1]]
                came_from_node = came_from.get(came_from_node)

            return np.array(path), total_cost

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

            if forbidden_map is not None and forbidden_map[neighbor[0], neighbor[1]] > 0:
                # this point is forbidden, no matter the cost, deny it
                continue

            if free_map is not None and free_map[neighbor[0], neighbor[1]] > 0:
                # this point is totally free to move
                cost = 0

            tentative_real_cost = real_costs[current] + cost_map[neighbor[0], neighbor[1]] + cost

            if neighbor not in openset:
                openset.add(neighbor)
            elif tentative_real_cost >= real_costs[neighbor]:
                continue

            # we found the best ever path
            came_from[neighbor] = current
            real_costs[neighbor] = tentative_real_cost
            heuristic_costs[neighbor] = real_costs[neighbor] + heuristic_fn(neighbor, ends)






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ms = plt.matshow

    # lets do some testing
    cost_map = np.ones((20,20))
    forbidden_map = np.zeros_like(cost_map)
    forbidden_map[10:20, 9] = 1
    forbidden_map[9, 0:8] = 1

    starts = [[0,0], [19,19]]
    ends = [[19,0], [19,1], [19,2], [19,3], [18,0], [18,1], [5,18]]
    starts = [0,0]
    ends = [5,18]


    path, cost = Astar_search(starts, ends, cost_map, heuristic_fn=closest_euclidian, forbidden_map=forbidden_map)

    visual = forbidden_map
    for p in path:
        visual[p[0],p[1]] = 2
    #  for p in ends:
        #  visual[p[0],p[1]] = 3
    #  for p in starts:
        #  visual[p[0],p[1]] = 4

    ms(visual)


