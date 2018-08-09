#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-26

import numpy as np
from scipy.spatial import Delaunay
import time
import sys

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
        else:
            if forces.shape != self._vel.shape:
                raise Exception('Force shape not the same as velocities! '+str(forces.shape))

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

        # planar obstacles to 'collide' off of
        # see the similarly named function
        # TODO yank this out into a function, doesnt need to be a method here
        self.planar_obstacles = []


    def add_planar_obstacle(self, p):
        """
        p represents a plane that cuts the target sphere. Said p is the 'middle' of the plane
        that when the normal of the plane is added to p, result is the center of the sphere.
        p + n = C.
        p is in xyz coords
        """
        self.planar_obstacles.append(p)



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
            # zero_out the penetrative if it is inside too
            zero_out = np.logical_or(zero_out, current_crossed)
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
            anti_penetrative_f = penetrative_f


            # pushing force is simply the plane's normal. how powerful it should be is another question.
            # simply the distance to plane is probably a good start
            pushing_f = -np.outer(current_distances ,n)
            # HACK4lyfe
            #  pushing_f += np.random.random(pushing_f.shape)*0.5
            #  pushing_f = np.outer([2]*N, n)
            for i in range(3):
                pushing_f[:,i] *= push_out

            # hacky, gives planes immense power to push out stuff
            # works though.
            pushing_f *= 500


            # penetrative_f and pushing_f should have mutually exclusive non-zero elements
            # so it is safe to sum them up
            # we need to negate the penetrative to make it 'anti-penetrative'
            # first add the 'zero-ing' forces
            forces += anti_penetrative_f
            # then add the pushing out forces where needed
            forces += pushing_f

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

        for i,this in enumerate(self._pos):
            # this is the normal vector on the sphere, we want the forces to be 0
            # on this vector so that the point does not try to move away from the sphere surface
            _, normal_vec = G.vec_normalize(this - self._center[i])
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


    def change_centers(self, c):
        """
        replace centers with the given one
        c needs to be the same shape as old center
        """
        c = np.array(c)
        assert c.shape == self._center.shape, "given new centers not the same shape as old!"
        self._center = c

    def change_radius(self, r):
        """
        replace the radius of the swarm with the given one
        new and old radius must have the same shape
        """
        r = np.array(r)
        assert r.shape == self._radius.shape, "given new radius not the same shape as old!"
        self._radius = r



if __name__=='__main__':
    ########################################################################################
    from rosview import RosSwarmView, RosPoseView, RosMarkerArrayView
    from dynamic_point import VelocityPoint
    import rospy

    # N, num of agents
    N = 10
    # dt, time step per sim tick
    # during run, the simulation will change between these when needed
    dt_steps = [0.005, 0.01, 0.05]
    current_dt_step = 0
    # ups, updates per second for ros stuff to update
    ups = 60
    # ticker per view, how many sim ticks to run per view update
    ticks_per_view = 10

    # set to false when not running a profiler
    # this changes the ros init_node disable signal and plotting stuff
    profiling = False

    # init the usual ros stuff
    rospy.init_node('rosviewtest', anonymous=True, disable_signals=profiling)

    rate = rospy.Rate(ups)

    # create the swarm with random starting positions
    init_pos = (np.random.random((N,3))*2)-1
    #  centers = [ (0,0,0), (1,0,0), (0,1,0), (0,0,1) ]
    #  radii = [ 0.2, 0.2, 0.2, 0.2]
    centers = [0,0,0]
    radii = 1
    # with some small mass and damping
    swarm = DynamicPointSphereSwarm(init_pos=init_pos,
                                    center=centers,
                                    radius=radii,
                                    mass=0.01,
                                    damping=0.05)

    spike_init_pos = (np.random.random((N,3))*2)-1
    spikes = DynamicPointSphereSwarm(init_pos=spike_init_pos,
                                     center=init_pos,
                                     radius=[0.4]*N,
                                     mass=0.01,
                                     damping=0.05)

    # we want fancy stuff to be shown, put them in a marker array
    # make the meshes slightly transparent
    marker_array = RosMarkerArrayView()

    # where the mesh files to display are stored for viewing
    mesh_dir = '/home/ozer/Dropbox/projects/toolbox/'

    ########################################################################################
    # SWARM INIT
    ########################################################################################
    # create the swarm view so we can see stuff in rviz
    swarm_view = RosSwarmView(swarm=swarm,
                              mesh_path=mesh_dir+'known_arrow.dae',
                              mesh_rgba=(0.2, 0.8, 0.2, 1),
                              mesh_scale=(1, 1, 1))

    spike_view = RosSwarmView(swarm=spikes,
                              mesh_path=mesh_dir+'known_arrow.dae',
                              mesh_rgba=(0.8, 0.8, 0.2, 1),
                              mesh_scale=(1, 1, 1),
                              last_used_id=swarm_view.last_used_id+10)

    # we also want to see a plane and sphere in rviz, representing the cage and obstacle
    # create views for them. Since these are stationary, we wont be updating them much
    # we just need the position of this, it wont be moving
    for center, radius in zip(swarm._center, swarm._radius):
        sphere_body = VelocityPoint(init_pos=center)
        sphere_view = RosPoseView(body=sphere_body,
                                  topic='/sphere_cage')
        marker_array.add_view(sphere_view, mesh_dir+'known_sphere.dae',
                              (radius,radius,radius),
                              (0.2,0.2,0.8,0.2))

    ########################################################################################
    # PLANES
    ########################################################################################
    planes = [ (0,0,0.8), (0,0,-0.8), (2,1,0.6)]#, (1.2, 1.2, -0.6)]
    #  planes = []
    plane_views = []
    plane_bodies = []
    for plane in planes:
        # a plane that is a little inside the sphere at 0,0,0 with r=1
        #  plane_pos = G.uvr_to_xyz((np.pi/6, np.pi/4, 0.5))
        plane_pos = G.uvr_to_xyz(plane)
        # the plane normal should be towards the sphere center at 0,0,0
        plane_normal = np.zeros_like(plane_pos) - plane_pos
        _, plane_normal = G.vec_normalize(plane_normal)

        # add the plane obstacle to the swarm
        swarm.add_planar_obstacle((plane_pos, plane_normal))
        spikes.add_planar_obstacle((plane_pos, plane_normal))

        # these points use the velocty for orientation
        plane_body = VelocityPoint(init_pos=plane_pos,
                                   init_vel=plane_normal)
        plane_view = RosPoseView(body=plane_body,
                                 topic='/plane_obstacle')
        plane_views.append(plane_view)
        plane_bodies.append(plane_body)
        marker_array.add_view(plane_view, mesh_dir+'known_plane.dae',(5,5,5),(0.8,0.2,0.2,0.5))

    ########################################################################################
    # INITIALIZE
    ########################################################################################
    # before we update, initialize the marker array
    marker_array.init()

    # update these for a little while until rviz realizes whats up and shows them
    rate2 = rospy.Rate(200)
    for i in range(100):
        # and show the markers
        for plane_view in plane_views:
            plane_view.update()
            #  plane_view2.update()
        sphere_view.update()
        marker_array.update()
        rate2.sleep()

    # to be plotted later
    sphere_err = []
    traces = []
    velocities = []
    edges = []
    applied_forces = []
    spike_applied_forces = []
    sphere_forces_over_time = []
    point_forces_over_time = []

    tick_times = []

    ########################################################################################
    # MAIN LOOP
    ########################################################################################
    # main sim loop
    while not rospy.is_shutdown():
        # update the physics of the swarm many times per view update
        for tick in range(ticks_per_view):
            # check the previous tick's applied forces and see if they are large or small
            # used to dynamically change the time step for every sim tick
            # to speed up those final ever-so-slightly-still-moving moments
            # all values eye-balled
            if len(applied_forces) > 0 and len(spike_applied_forces) > 0:
                last_max_force = max( np.max(np.abs(applied_forces[-1])),
                                      np.max(np.abs(spike_applied_forces[-1])))
                if last_max_force > 0.01:
                    current_dt_step = 0
                elif last_max_force <= 0.01:
                    current_dt_step = 1
                    if last_max_force <= 0.005:
                        current_dt_step = 2

            # update dt
            dt = dt_steps[current_dt_step]

            t0 = time.time()

            # update the swarm
            sphere_forces = swarm.calc_sphere_forces(dt)
            point_forces = swarm.calc_point_forces(dt)
            point_forces *= 0.05
            forces = sphere_forces+point_forces
            forces = swarm.handle_obstacle_forces(dt, forces)
            swarm.update(dt, forces=forces)

            # same for the spikes
            # but also update the centers of the spikes so that they
            # follow the swarm
            spikes.change_centers(swarm._pos)
            s_sphere_forces = spikes.calc_sphere_forces(dt)
            s_point_forces = spikes.calc_point_forces(dt)
            s_point_forces *= 0.05
            s_forces = s_sphere_forces + s_point_forces
            s_forces = spikes.handle_obstacle_forces(dt, s_forces)
            spikes.update(dt, forces=s_forces)

            tick_times.append(time.time()-t0)

            # record for later
            edges.append(swarm.cage_status())
            traces.append(np.copy(swarm._pos))
            sphere_err.append(swarm.check_sphere())
            applied_forces.append(forces)
            sphere_forces_over_time.append(sphere_forces)
            point_forces_over_time.append(point_forces)
            velocities.append(np.copy(swarm._vel))

            spike_applied_forces.append(s_forces)

        # finally show the state of the swarm
        swarm_view.update()
        spike_view.update()
        rate.sleep()

        # stop if agents converged
        if not swarm._any_alive() and not spikes._any_alive():
            print('All dead. Avg ticks per second:', 1/np.average(tick_times))
            print('Ticks:',len(applied_forces))
            break

    ########################################################################################
    # FINALIZE
    ########################################################################################
    # send the final locations to rviz
    for i in range(50):
        swarm_view.update()
        spike_view.update()
        rate2.sleep()

    # no need to profile plotting
    if profiling:
        sys.exit(0)
    ########################################################################################
    # PLOTTING
    ########################################################################################
    # time, agent, (x,y,z)
    traces = np.array(traces)
    # time, edge, [(x,y,z), (x,y,z)]
    edges_in_time = np.array(edges)
    # time, agent, (u,v,w)
    applied_forces = np.array(applied_forces)
    point_forces_over_time = np.array(point_forces_over_time)
    sphere_forces_over_time = np.array(sphere_forces_over_time)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #needed for '3d'
    plt.ion()
    import plotting as P

    ########################################################################################
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(sphere_err)
    plt.xlabel('sim ticks')
    plt.ylabel('sphere err')
    plt.title('sphere error')

    edge_lens = []
    for t,edges in enumerate(edges_in_time):
        for edge in edges:
            if edge is not None:
                L = G.euclid_distance(edge[0], edge[1])
                edge_lens.append((t,L))
    edge_lens = np.array(edge_lens)

    plt.subplot(1,2,2)
    # edge_lens might not have the same number of edges at every time step!
    # so we can not really make a 2d array out of it to plot
    # we can, however, scatter what we got
    plt.scatter(edge_lens[:,0], edge_lens[:,1], marker='.', alpha=0.3)
    plt.axhline(1.63, linestyle='--')
    plt.xlabel('sim ticks')
    plt.ylabel('edge lengths of cage')
    plt.ylim(0,3)
    plt.title('cage edges over time')
    ########################################################################################

    fig, ax = P.make_3d_fig()
    P.scatter3(ax, swarm._pos)
    P.scatter3(ax, np.array((0,0,0)), color='g')

    every=200
    for i in range(N):
        P.plot3(ax, traces[::every][:,i,:])
        P.arrows3(ax, traces[::every][:,i,:], point_forces_over_time[::every][:,i,:], length=0.1, normalize=True, color='r', alpha=0.3)
        P.arrows3(ax, traces[::every][:,i,:], sphere_forces_over_time[::every][:,i,:], length=0.1, normalize=True, color='r', alpha=0.3)

    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    plt.title('trajectories of points')




    ########################################################################################
    # plot a sphere, check each point for coverage
    nu, nv = 100,100
    sensor_range = 0.5
    uu, vv = np.meshgrid(np.linspace(0,np.pi,nu), np.linspace(-np.pi, np.pi, nv))
    colors = []
    xyzs = []
    sp_uvs = []
    sp_colors = []
    for i in range(nu):
        for j in range(nv):
            xyz = G.uvr_to_xyz((uu[i,j],vv[i,j],1))
            # project point to the planes to see if it is 'outside' any
            distance = None
            for A,n in swarm.planar_obstacles:
                projection, distance = G.project_point_to_plane(xyz, A, n)
                if distance < 0:
                    xyz = projection
                    break

            xyzs.append(xyz)
            sp_uvs.append(G.xyz_to_uvr(xyz))

            # color the planes differently
            if distance is not None and distance<0:
                colors.append(0.5)
                sp_colors.append(0.5)
                continue

            dist = G.euclid_distance(xyz, swarm._pos)
            mindist = np.min(dist)
            if mindist<sensor_range:
                colors.append(1)
                sp_colors.append(1)
            else:
                colors.append(0)
                sp_colors.append(0)

    # normalize colors to 0-1
    colors = np.array(colors, dtype='float64')
    maxdist = np.max(colors)
    colors /= maxdist

    fig, ax = P.make_3d_fig()
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99)
    ax.set_xlim(left=-1,right=1)
    ax.set_ylim(bottom=-1,top=1)
    ax.set_zlim(bottom=-1,top=1)
    # move the agents a little outside the sphere so we can see them
    pts = G.xyz_to_uvr(swarm._pos)
    pts[:,2] += 0.05
    pts = G.uvr_to_xyz(pts)
    P.scatter3(ax, pts, c='r')
    c = P.surface_tri(ax, xyzs, uu, vv, colors, cmap='seismic', linewidth=0, alpha=0.7)

    #############################################
    # spherical coords plot
    #############################################
    sp_colors = np.array(sp_colors)
    sp_uvs = np.array(sp_uvs)[:,:2]
    plt.figure()
    plt.scatter(sp_uvs[:,0], sp_uvs[:,1], c=sp_colors)
    plt.title('uv-coords version')



