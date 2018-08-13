#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-09

import numpy as np
import geometry as G

def collide_planes(obj, forces, planar_obstacles, dt):
    """
    given some forces currently acting on some points, return forces that wont make points
    go through obstacles.

    obj needs to have a get_position() function that returns (N,3) array
    also a set_position(pos) function that directly sets the positions
    same goes for velocity, get_velocity and set_velocity

    and an update(dt, forces) function that will modify the positions of the object given the forces

    forces is expected to be (N,3) vectors acting on obj

    planar_obstacles to be [ (A,n), ...] where
    A is a point on a plane and n is the normal vector of the plane

    dt is time that will pass while applying the forces returned from this function
    """
    pos = obj.get_position()
    vel = obj.get_velocity()
    N = pos.shape[0]
    # record these so that we can put them back at the end, the mock updates 
    # will modify these otherwise
    current_pos = np.copy(pos)
    current_vel = np.copy(vel)

    # we need to find the part of the forces that will move the points
    # over to the opposite side of the plane
    # so first we need to check if the current forces will push the point
    # over to the other side of the plane
    # this is a "mock-update"
    obj.update(dt, forces)
    # now we are in 1 frame in the future

    future_pos = obj.get_position()

    for A,n in planar_obstacles:
        # project the current points to see if we ALREADY are crossed
        current_projections, current_distances = G.project_point_to_plane(current_pos, A, n)
        # project our points in the future to the plane to see if we cross over next frame
        future_projections, future_distances = G.project_point_to_plane(future_pos, A, n)
        # if distance to plane < 0, then we are on the 'wrong' side of it
        # we will use this mask to decide which points will have their forces nerfed

        current_crossed = current_distances < 0
        future_crossed = future_distances < 0
        current_not_crossed = np.logical_not(current_crossed)

        # if point is currently crossed and crossed in the future, push it out
        push_out = np.logical_and(current_crossed, future_crossed)
        # if point is currently not crossed but crossed in the future, zero the penetrative forces
        zero_out = np.logical_and(current_not_crossed, future_crossed)
        # zero_out the penetrative if it is inside too
        zero_out = np.logical_or(zero_out, current_crossed)

        # now we have 3 masks that filter out the three possible actions we can take on points

        # project the forces to the plane's normal, this gives us the penetrative
        # part of the forces acting on the points, we want to have 0 penetrative force
        # on the points that will pass over, given by mask.
        penetrative_f = G.project_vec(forces, n)
        # mask out the points that wont pass over
        for i in range(3):
            penetrative_f[:,i] *= zero_out
        anti_penetrative_f = penetrative_f


        # pushing force is simply the plane's normal. how powerful it should be is another question.
        # simply the distance to plane is probably a good start
        pushing_f = -np.outer(current_distances ,n)
        for i in range(3):
            pushing_f[:,i] *= push_out

        # hacky, gives planes immense power to push out stuff
        # works though.
        pushing_f *= 100


        # penetrative_f and pushing_f should have mutually exclusive non-zero elements
        # so it is safe to sum them up
        # we need to negate the penetrative to make it 'anti-penetrative'
        # first add the 'zero-ing' forces
        forces += anti_penetrative_f
        # then add the pushing out forces where needed
        forces += pushing_f

    # reset to real current values
    obj.set_position(current_pos)
    obj.set_velocity(current_vel)

    return forces


def apply_chain(head, tail, length, head_forces, tail_forces, dt):
    """
    head<><><><>tail
    tail should not be able to move more than length away from head.

    This is done by checking the next frame for this distance.
    If in the next frame tail is farther than allowed, forces are modified so that it
    will not be farther.

    head and tail should have get_position, get_velocity, set_position, set_velocity.
    get's should return same shape for both
    both should also have update(forces, dt)
    same stuff as plane collisions really
    """

    current_head_pos = head.get_position()
    current_head_vel = head.get_velocity()

    current_tail_pos = tail.get_position()
    current_tail_vel = tail.get_velocity()

    head.update(dt, forces=head_forces)
    tail.update(dt, forces=tail_forces)

    future_head_pos = head.get_position()
    future_tail_pos = tail.get_position()

    future_distances = G.euclid_distance(future_tail_pos, future_head_pos)
    # a mask of points where in the future the chain will break
    future_breaks = future_distances > length
    # to make it from a (N,) to (N,1) so we can do broadcasting stuff later
    future_breaks = np.transpose(np.atleast_2d(future_breaks))

    # vector from tail to head is the 'chain'
    current_chains = current_head_pos - current_tail_pos
    # project the force on tail on the 'chain'
    chain_pulls = [G.project_vec(tail_forces[i], current_chains[i]) for i in range(len(tail_forces)) ]
    chain_pulls = np.array(chain_pulls)*future_breaks
    # subtract these pulls from the acting forces on the tail
    non_pulling_forces = tail_forces - chain_pulls

    # undo the updates
    head.set_position(current_head_pos)
    head.set_velocity(current_head_vel)
    tail.set_position(current_tail_pos)
    tail.set_velocity(current_tail_vel)

    return non_pulling_forces
