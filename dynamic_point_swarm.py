#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-26

import numpy as np
from scipy.spatial import Delaunay

import geometry as G
import Pid
from Quaternion import Quat

RADTODEG = 360 / (np.pi * 2)

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

    def cage_status(self):
        """
        return the cage edges and distances between any two neighbor
        """

        # triangulate the points and generate a convex hull out of it
        # for well-formed spherical-formations, this convex hull is convex and approximates
        # the sphere.
        try:
            tris = Delaunay(self._pos)
        except:
            # if can't triangulate, the vertices must be degenerate, so no cage.
            return None, None
        # hull is made of triangle edges in terms of indices for self._pos
        hull = tris.convex_hull

        edges = {}
        for triangle in hull:
            # for each triangle, we want to know the edge lengths
            # make the triangle a loop by adding the first vertex at the end
            poly = list(triangle) + [triangle[0]]
            for i in range(1, len(poly)):
                i1 = poly[i-1]
                i2 = poly[i]
                v1 = self._pos[i1]
                v2 = self._pos[i2]
                # i only want unique edges, dont give me '0:2 and 2:0' BS.
                if i1 > i2:
                    edge_key = str(i1)+':'+str(i2)
                else:
                    edge_key = str(i2)+':'+str(i1)
                edges[edge_key] = (v1,v2)

        return np.array(list(edges.values()))

    def update(self, dt, forces=None, return_positions=False):
        """
        dt is time spent for this update in seconds
        forces is an np array of size (N,3)

        if return_positions is True, returns a copy of the position array
        """

        if forces is None:
            forces = np.zeros((self._num_points,3))
        else:
            if forces.shape != self._vel.shape:
                raise Exception('Force shape not the same as velocities! '+str(forces.shape))


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

    def get_positions(self):
        return self._pos

    def get_orientation_quats(self):
        # use the velocity as orientation
        # ra=heading/yaw, dec=pitch, roll=roll
        quats =[]
        for vel in self._vel:
            yaw,pitch = G.vec3_to_yaw_pitch(vel)
            roll = 0

            yaw*=RADTODEG
            pitch*=RADTODEG
            quats.append(Quat([yaw,pitch,roll]).q)

        return quats




class DynamicPointSphereSwarm(DynamicPointSwarm):
    def __init__(self,
                 center=[0,0,0],
                 radius=1,
                 charge=1,
                 max_surf_error = 0.3,
                 **kwargs):
        """
        a swarm of points in 3D space with a soft restriction of staying
        on the given sphere
        """
        super().__init__(**kwargs)

        self._center = np.array(center, dtype='float64')
        self._radius = radius
        self._charge = charge

        # if away from the sphere this much, points will ignore others' push
        self._max_surf_error = max_surf_error

        # PID values determined experimentally
        P = 2
        I = 0.01
        D = 0.5
        self._PIDs = [Pid.PID(P,I,D) for i in self._pos]

        # planar obstacles to 'collide' off of
        # see the similarly named function
        self.planar_obstacles = []


    def add_planar_obstacle(self, p):
        """
        p represents a plane that cuts the target sphere. Said p is the 'middle' of the plane
        that when the normal of the plane is added to p, result is the center of the sphere.
        p + n = C.
        p is in xyz coords
        """
        self.planar_obstacles.append(p)

    def handle_plane_collisions(self):
        """
        'push' the points to the plane such that they are at the side of the plane where the normal
        vector is pointing towards
        """

        for A,n in self.planar_obstacles:
            # project the points on the plane, if dist < 0 then the point
            # needs to be moved to the projected point
            projections, distances = G.project_point_to_plane(self._pos, A, n)
            # difference from current positions to the projected positions
            # we want to 0 out the difference where the point is on the 'right'
            # side of the plane
            # dX is towards the planes normal
            dX = projections - self._pos
            # this creates a boolean array where true=needs to be moved
            mask = distances < 0
            # mask is (N,) dX is (N,3), we want to create an
            # (N,3) movement array that has (0,0,0) where mask=False
            # basically multiply all rows with mask
            for i in range(3):
                dX[:,i] *= mask
            self._pos += dX

        # will break when more than one plane !!!!!!!!!!!!!!
        return distances


    def handle_obstacle_forces(self, dt, forces):
        """
        given some forces currently acting on points, return forces that wont make points
        go through obstacles.
        """
        N = self._pos.shape[0]
        # record these so that we can put them back at the end, the mock updates 
        # will modify these otherwise
        current_real_pos = np.copy(self._pos)
        current_real_vel = np.copy(self._vel)

        # we need to find the part of the forces that will move the points
        # over to the opposite side of the plane
        # so first we need to check if the current forces will push the point
        # over to the other side of the plane
        # this is a "mock-update"
        self.update(dt, forces)
        # now we are in 1 frame in the future

        for A,n in self.planar_obstacles:
            # project the current points to see if we ALREADY are crossed
            current_projections, current_distances = G.project_point_to_plane(current_real_pos, A, n)
            # project our points in the future to the plane to see if we cross over next frame
            future_projections, future_distances = G.project_point_to_plane(self._pos, A, n)
            # if distance to plane < 0, then we are on the 'wrong' side of it
            # we will use this mask to decide which points will have their forces nerfed

            current_crossed = current_distances < 0
            future_crossed = future_distances < 0

            current_not_crossed = np.logical_not(current_crossed)
            future_not_crossed = np.logical_not(future_crossed)

            # if point is currently crossed but not crossed in the future, do nothing to it
            do_nothing = np.logical_and(current_crossed, future_not_crossed)
            # if point is currently crossed and crossed in the future, push it out
            push_out = np.logical_and(current_crossed, future_crossed)
            # if point is currently not crossed but crossed in the future, zero the penetrative forces
            zero_out = np.logical_and(current_not_crossed, future_crossed)
            # if point is currently not crossed and not crossed in the future, do nothing to it either
            do_nothing = np.logical_or(do_nothing, np.logical_and(current_not_crossed, future_not_crossed))

            # now we have 3 masks that filter out the three possible actions we can take on points

            # project the forces to the plane's normal, this gives us the penetrative
            # part of the forces acting on the points, we want to have 0 penetrative force
            # on the points that will pass over, given by mask.
            penetrative_f = G.project_vec(forces, n)
            # mask out the points that wont pass over
            for i in range(3):
                penetrative_f[:,i] *= zero_out

            # pushing force is simply the plane's normal. how powerful it should be is another question.
            # simply the distance to plane is probably a good start
            #  pushing_f = np.outer(current_distances ,n)
            pushing_f = np.outer([2]*N, n)
            for i in range(3):
                pushing_f[:,i] *= push_out

            print('--')
            print(penetrative_f)
            print(pushing_f)
            # penetrative_f and pushing_f should have mutually exclusive non-zero elements
            # so it is safe to sum them up
            # we need to negate the penetrative to make it 'anti-penetrative'
            net_f = -penetrative_f*10 + pushing_f*5
            forces += net_f




        # reset to real current values
        self._pos = current_real_pos
        self._vel = current_real_vel

        return forces

    def check_sphere(self):
        """
        return the distances to the sphere's surface for all points in swarm
        """

        dists = G.euclid_distance(self._pos, self._center)
        errs = dists - self._radius
        return errs


    def calc_forces(self, dt):
        """
        calc the net force acting on all points.
        """

        forces = np.zeros_like(self._pos)
        tangent_forces = np.zeros_like(self._pos)

        surface_errors = self.check_sphere()

        for i,this in enumerate(self._pos):
            surface_error = surface_errors[i]

            # this is the normal vector on the sphere, we want the forces to be 0
            # on this vector so that the point does not try to move away from the sphere surface
            _, normal_vec = G.vec_normalize(this - self._center)
            # we always want the point to go towards a point that is ON the sphere.
            # if it is somehow away from the surface, add a force that will push it towards it.
            correction = self._PIDs[i].update(surface_error, dt)
            forces[i] += normal_vec * correction

            # do not apply forces from others onto this particle if it is out of the sphere
            # surface. This makes it so the particles prioritize surface before distribution.
            if np.abs(surface_error) > self._max_surf_error:
                continue


            # add up all the other points' effects on this point.
            for j,other in enumerate(self._pos):
                dist = G.euclid_distance(this, other)
                if dist < 0.01:
                    # disregards self and too-close points
                    continue

                # magnitude of the force is not the vectors norm!
                force_mag = self._charge / (dist**4)
                _, force_vec = G.vec_normalize(this - other)
                force_vec *= force_mag
                perpendicular_vec = G.project_vec(force_vec, normal_vec)
                tangent_vec = force_vec - perpendicular_vec
                tangent_forces[i] += tangent_vec

        # finally add the tangent forces, if any, to the suface forces
        # added previously
        forces += tangent_forces
        return forces







if __name__=='__main__':
    from rosview import RosSwarmView, RosPoseView, RosMarkerArrayView
    from dynamic_point import VelocityPoint
    import rospy

    # N, num of agents
    N = 4
    # dt, time step per sim tick
    dt = 0.005
    # ups, updates per second for ros stuff to update
    ups = 90
    # ticker per view, how many sim ticks to run per view update
    ticks_per_view = 1

    # init the usual ros stuff
    rospy.init_node('rosviewtest', anonymous=True)
    rate = rospy.Rate(ups)

    # create the swarm with random starting positions
    init_pos = np.random.random((N,3))
    # with some small mass and damping
    swarm = DynamicPointSphereSwarm(init_pos=init_pos,
                                    mass=0.01,
                                    damping=0.1)

    # where the mesh files to display are stored for viewing
    mesh_dir = '/home/ozer/Dropbox/projects/toolbox/'

    # create the swarm view so we can see stuff in rviz
    swarm_view = RosSwarmView(swarm=swarm,
                              mesh_path=mesh_dir+'known_arrow.dae',
                              mesh_rgba=(0.2, 0.8, 0.2, 1),
                              mesh_scale=(1, 1, 1))

    # we also want to see a plane and sphere in rviz, representing the cage and obstacle
    # create views for them. Since these are stationary, we wont be updating them much
    # we just need the position of this, it wont be moving
    sphere_body = VelocityPoint(init_pos=(0, 0, 0))
    sphere_view = RosPoseView(body=sphere_body,
                              topic='/sphere_cage')


    # a plane that is a little inside the sphere at 0,0,0 with r=1
    #  plane_pos = G.uvr_to_xyz((np.pi/6, np.pi/4, 0.5))
    plane_pos = G.uvr_to_xyz((0, 0, 0.5))
    # the plane normal should be towards the sphere center at 0,0,0
    plane_normal = np.zeros_like(plane_pos) - plane_pos

    # add the plane obstacle to the swarm
    swarm.add_planar_obstacle((plane_pos, plane_normal))

    # these points use the velocty for orientation
    plane_body = VelocityPoint(init_pos=plane_pos,
                               init_vel=plane_normal)
    plane_view = RosPoseView(body=plane_body,
                             topic='/plane_obstacle')

    # we want fancy stuff to be shown, put them in a marker array
    # make the meshes slightly transparent
    marker_array = RosMarkerArrayView(ros_pose_viewers=[sphere_view,
                                                        plane_view],
                                      mesh_paths = [mesh_dir+'known_sphere.dae',
                                                    mesh_dir+'known_plane.dae'],
                                      mesh_rgbas = [(0.2, 0.2, 0.8, 0.2),
                                                    (0.8, 0.2, 0.2, 0.5)],
                                      mesh_scales = [(1, 1, 1),
                                                     (5, 5, 5)])



    # update these for a little while until rviz realizes whats up and shows them
    rate2 = rospy.Rate(200)
    for i in range(100):
        # and show the markers
        plane_view.update()
        sphere_view.update()
        marker_array.update()
        rate2.sleep()

    # to be plotted later
    sphere_err = []
    traces = []
    edges = []

    # main sim loop
    while not rospy.is_shutdown():
        # update the physics of the swarm many times per view update
        for tick in range(ticks_per_view):
            forces = swarm.calc_forces(dt)
            #  forces = swarm.handle_plane_collisions()
            forces = swarm.handle_obstacle_forces(dt, forces)
            swarm.update(dt, forces=forces)
            # record for later
            edges.append(swarm.cage_status())
            traces.append(np.copy(swarm._pos))
            sphere_err.append(swarm.check_sphere())
        # finally show the state of the swarm
        swarm_view.update()

        rate.sleep()

        # stop if agents converged
        if not swarm._any_alive():
            print('all dead')
            break

    # send the final locations to rviz
    for i in range(20):
        swarm_view.update()
        rate2.sleep()

    # time, agent, (x,y,z)
    traces = np.array(traces)
    # time, edge, [(x,y,z), (x,y,z)]
    edges_in_time = np.array(edges)
    # time, agent
    #  collisions = np.array(collisions)

####################################################################################################################
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #needed for '3d'
    plt.ion()

    #  plt.figure()
    #  plt.plot(sphere_err)
    #  plt.xlabel('sim ticks')
    #  plt.ylabel('sphere err')
    #  plt.title('sphere error')

    edge_lens = []
    for t,edges in enumerate(edges_in_time):
        time_lens = []
        for edge in edges:
            L = G.euclid_distance(edge[0], edge[1])
            time_lens.append(L)
        edge_lens.append(time_lens)
    print('final edge lens:',edge_lens[-1])
    plt.figure()
    edge_lens = np.array(edge_lens)
    for i in range(edge_lens.shape[1]):
        plt.plot(edge_lens[:,i])
    plt.axhline(1.63, linestyle='--')
    plt.xlabel('sim ticks')
    plt.ylabel('edge lengths of cage')
    plt.title('cage edges over time')

    fig = plt.figure(figsize=(10,10))
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    ax.azim=45
    ax.elev=30
    ax.scatter(swarm._pos[:,0], swarm._pos[:,1], swarm._pos[:,2])
    ax.scatter(plane_pos[0], plane_pos[1], plane_pos[2], color='r')
    ax.scatter([0],[0],[0], color='g')

    for i in range(N):
        ax.plot3D(traces[:,i,0],traces[:,i,1],traces[:,i,2])

    plt.title('trajectories of points')



