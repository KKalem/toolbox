#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-05-24

import numpy as np

try:
    # explicit relative imports with dots in front.
    from .Pid import PID
    from .Quaternion import Quat
    from . import geometry as geom
except SystemError:
    # absolute imports when we want to run this by itself
    from Pid import PID
    from Quaternion import Quat
    import geometry as geom

RADTODEG = 360 / (np.pi * 2)


class DynamicPoint:
    def __init__(self,
                 mass=1,
                 init_pos=(0,0,0),
                 init_vel = (0,0,0),
                 init_acc = (0,0,0),
                 damping=None,
                 max_vel=None,
                 max_acc=None
                 ):

        self.mass = mass
        self.pos = np.array(init_pos, dtype='float64')
        self.vel = np.array(init_vel, dtype='float64')
        self.acc = np.array(init_acc, dtype='float64')
        self.damping = damping
        self.max_vel = max_vel
        self.max_acc = max_acc



    def _limit_vel(self, vel):
        if self.max_vel is not None:
            vel = geom.vec_limit_len(vel, self.max_vel)
        return vel

    def _limit_acc(self, acc):
        if self.max_acc is not None:
            acc = geom.vec_limit_len(acc, self.max_acc)
        return acc

    def _apply_damping(self):
        if self.damping is not None:
            self.vel -= self.vel*self.damping

    def update(self, dt):
        self.vel += self.acc * dt
        self.vel = self._limit_vel(self.vel)
        self.pos += self.vel * dt

    def get_position(self):
        return self.pos

    def get_orientation_quat(self):
        # use the velocity as orientation
        # ra=heading/yaw, dec=pitch, roll=roll
        yaw,pitch = geom.vec3_to_yaw_pitch(self.vel)
        roll = 0

        yaw*=RADTODEG
        pitch*=RADTODEG

        return Quat([yaw,pitch,roll]).q



class VelocityPoint(DynamicPoint):
    def __init__(self, speed=None, **kwargs):
        super().__init__(**kwargs)

        self.target = DynamicPoint()
        if speed is None:
            self.speed = 999999
        else:
            self.speed = speed


    def set_target(self, target):
        self.target.pos = np.array(target, dtype='float64')

    def update(self, dt):
        if self.target is not None:
            target_dist, target_vec = geom.vec_normalize(self.target.get_position() - self.pos)
            if target_dist > 0.1:
                self.vel = target_vec * self.speed
            else:
                self.vel = np.zeros(3)

        self.vel = self._limit_vel(self.vel)
        self.pos += self.vel * dt




class PIDPoint(DynamicPoint):
    def __init__(self,
                 pid=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.target = DynamicPoint()
        if pid is None:
            self.pid = PID(P=0.2, D=0.5)
        else:
            self.pid = PID(*pid)

    def set_target(self, target):
        self.target.pos = np.array(target, dtype='float64')
        self.pid.clear()

    def update(self, dt):
        self.acc = np.array((0.,0.,0.))
        if self.target is not None:
            target_dist, target_vec = geom.vec_normalize(self.target.get_position() - self.pos)
            if target_dist > 0:
                forward_acc = self.pid.update(target_dist, dt)
                self.acc -= forward_acc * target_vec

        self._apply_damping()

        self.acc = self._limit_acc(self.acc)
        self.vel += self.acc * dt

        self.vel = self._limit_vel(self.vel)
        self.pos += self.vel * dt


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    p = PIDPoint()
    p.set_target((5,5,5))

    trace = []
    for i in range(10000):
        p.update(0.01)
        trace.append(p.pos.copy())

    p.set_target((0,0,0))
    for i in range(10000):
        p.update(0.01)
        trace.append(p.pos.copy())

    trace = np.array(trace)

    plt.plot(range(len(trace)),trace[:,0])

