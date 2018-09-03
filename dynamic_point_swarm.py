#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-07-26

import numpy as np
import time
import sys

from . import geometry as G
from . import Pid
from .Quaternion import Quat

class DynamicPointSwarm:
    def __init__(self,
                 init_pos=None,
                 num_agents=4,
                 init_vel=None,
                 mass=1,
                 damping=0):
        """
        A collection of points moving about in 3D space with no restrictons

        init_pos and init_vel are list of [x,y,z] or np array of shape (N,3)
        if num_agents is given with init_pos, init_pos is used.
        if nothing is given, 4 agents at 0,0,0 will be created.
        """

        if init_pos is not None:
            # check and assign the initial positions for the particles
            if init_pos.shape[1] != 3:
                raise Exception('Illegal init_pos shape to DynamicSwarm3. Expected (N,3), got '+str(init_pos.shape[1]))

            self._pos = init_pos
        else:
            self._pos = np.zeros((num_agents,3))

        self._num_points = self._pos.shape[0]

        # if given, assign the velocities too, otherwise assume 0
        if init_vel is not None:
            if init_vel.shape[1] != 3:
                raise Exception('Illegal init_vel shape to DynamicSwarm3. Expected (N,3), got '+str(init_vel.shape[1]))
            # given vels
            self._pos = init_pos
        else:
            # not given vels
            self._vel = np.zeros_like(self._pos)

        # all particles have the same mass and damping
        self._mass = mass
        self._damping = damping

    def _any_alive(self, vel_limit=0.01):
        """
        if no particle is moving, retun false, otherwise return true
        """

        alives = np.abs(self._vel) >= vel_limit
        return np.sum(alives) > 1


    def update(self, dt, forces=None, return_positions=False, max_force_allowed=10):
        """
        dt is time spent for this update in seconds
        forces is an np array of size (N,3)

        if return_positions is True, returns a copy of the position array
        max_force_allowed clips the applied forces to +- that value so that the points do not
        fly off into the sun.
        """

        if forces is None:
            forces = np.zeros_like(self._pos)
        assert forces.shape == self._vel.shape,\
               'Force shape not the same as velocities! '+str(forces.shape)

        # prevent sun visits
        np.clip(forces, -10, 10, out=forces)
        # apply simple damping
        self._vel *= (1 - self._damping)
        # F = ma
        acc = forces/self._mass
        # v = at
        self._vel += acc*dt
        # x = vt
        self._pos += self._vel*dt

        if return_positions:
            return np.copy(self._pos)

    def get_position(self):
        return self._pos

    def set_position(self, pos):
        pos = np.array(pos)
        assert pos.shape == self._pos.shape, 'Given position different shape from old one!'
        self._pos = pos

    def get_velocity(self):
        return self._vel

    def set_velocity(self, vel):
        vel = np.array(vel)
        assert vel.shape == self._vel.shape, 'Given velocity different shape from old one!'
        self._vel = vel

    def get_orientation_quat(self):
        # use the velocity as orientation
        # ra=heading/yaw, dec=pitch, roll=roll
        quats =[]
        for vel in self._vel:
            yaw,pitch = G.vec3_to_yaw_pitch(vel)
            roll = 0

            yaw*= G.RADTODEG
            pitch*= G.RADTODEG
            quats.append(Quat([yaw,pitch,roll]).q)

        return quats


