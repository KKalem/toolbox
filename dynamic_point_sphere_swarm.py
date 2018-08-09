#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-07-26

import numpy as np


import geometry as G
import Pid
from Quaternion import Quat
from dynamic_point_swarm import DynamicPointSwarm

class DynamicPointSphereSwarm(DynamicPointSwarm):
    def __init__(self,
                 center=None,
                 radius=None,
                 charge=1,
                 max_surf_error = 0.3,
                 **kwargs):
        """
        a swarm of points in 3D space with a soft restriction of staying
        on the given sphere

        center can be a single (x,y,z) or (N,3) array
        radius can be a single value or (N,) array
        """
        super().__init__(**kwargs)

        N = self._pos.shape[0]

        if center is None:
            self._center = np.zeros_like(self._pos, dtype='float64')
        else:
            self._center = np.array(center, dtype='float64')
            try:
                # we were given a center per point
                assert self._center.shape == self._pos.shape, "Given center shape is not same as positions shape!"
            except:
                # we were not given a center per point, a single center for all of them then?
                self._center = np.atleast_2d(self._center)
                self._center = np.vstack([center]*N)
                assert self._center.shape == self._pos.shape, "Given center shape is not same as positions shape!"

        if radius is None:
            self._radius = np.ones_like(self._pos)
        else:
            self._radius = np.array(radius, dtype='float64')
            try:
                # enough radii for centers?
                assert self._radius.shape[0] == self._center.shape[0]
            except:
                # nope, apparently not
                # single radius?
                if self._radius.shape == ():
                    # this is only true if radius is a single number and not a list or sth. like it
                    # we need a radius per center, even if they are the same
                    self._radius = np.hstack([radius]*N)
                assert self._radius.shape == (N,), "Radius is f'ed up yo"


        self._center = np.array(self._center, dtype='float64')
        self._radius = np.array(self._radius, dtype='float64')
        self._charge = charge

        # if away from the sphere this much, points will ignore others' push
        self._max_surf_error = max_surf_error

        # PID values determined experimentally
        P = 2
        I = 0.01
        D = 0.5
        self._PIDs = [Pid.PID(P,I,D) for i in self._pos]


    def check_sphere(self):
        """
        return the distances to the sphere's surface for all points in swarm
        """

        dists = G.euclid_distance(self._pos, self._center)
        errs = dists - self._radius
        return errs


    def calc_sphere_forces(self, dt):
        """
        calc the net force acting on all points.
        """

        forces = np.zeros_like(self._pos)
        surface_errors = self.check_sphere()

        for i,this in enumerate(self._pos):
            surface_error = surface_errors[i]
            center = self._center[i]
            pid = self._PIDs[i]

            # this is the normal vector on the sphere, we want the forces to be 0
            # on this vector so that the point does not try to move away from the sphere surface
            _, normal_vec = G.vec_normalize(this - center)
            # we always want the point to go towards a point that is ON the sphere.
            # if it is somehow away from the surface, add a force that will push it towards it.
            correction = pid.update(surface_error, dt)
            forces[i] += normal_vec * correction

            # do not apply forces from others onto this particle if it is out of the sphere
            # surface. This makes it so the particles prioritize surface before distribution.
            if np.abs(surface_error) > self._max_surf_error:
                continue

        return forces

    def calc_point_forces(self, dt):
        tangent_forces = np.zeros_like(self._pos)
        _, normals = G.vec_normalize(self._pos - self._center)

        for i,this in enumerate(self._pos):
            # this is the normal vector on the sphere, we want the forces to be 0
            # on this vector so that the point does not try to move away from the sphere surface
            normal_vec = normals[i]
            # add up all the other points' effects on this point.
            for j,other in enumerate(self._pos):
                dist = G.euclid_distance(this, other)
                if dist < 0.05:
                    # disregards self and too-close points
                    continue

                # magnitude of the force is not the vectors norm!
                force_mag = self._charge / (dist**2)
                _, force_vec = G.vec_normalize(this - other)
                force_vec *= force_mag

                perpendicular_vec = G.project_vec(force_vec, normal_vec)
                tangent_vec = force_vec - perpendicular_vec
                tangent_forces[i] += tangent_vec

        return tangent_forces


    def get_acting_forces(self, dt):
        sf = self.calc_sphere_forces(dt)
        pf = self.calc_point_forces(dt)
        return sf + 0.05*pf

    def set_center(self, c):
        """
        replace centers with the given one
        c needs to be the same shape as old center
        """
        c = np.array(c)
        assert c.shape == self._center.shape, "given new centers not the same shape as old!"
        self._center = c

    def set_radius(self, r):
        """
        replace the radius of the swarm with the given one
        new and old radius must have the same shape
        """
        r = np.array(r)
        assert r.shape == self._radius.shape, "given new radius not the same shape as old!"
        self._radius = r

