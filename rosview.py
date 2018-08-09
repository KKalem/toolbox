#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-05-24

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


class RosPoseView:
    def __init__(self,
                 body,
                 topic,
                 mesh_path=None,
                 mesh_scale=(1,1,1),
                 mesh_rgba=(0,0,0,1)):

        self.body = body
        self.pub = rospy.Publisher(topic, PoseStamped, queue_size=1)


        # prepare the published stuff
        # these will simply be updated and not re-created
        point = Point()
        quat = Quaternion()
        pose = Pose()
        pose.position = point
        pose.orientation = quat
        self.pose = pose
        self.pose_stamped = PoseStamped()
        self.pose_stamped.header.frame_id = '/world'

        # start with the current state of the body
        self._update_pose()


    def _update_pose(self):
        x,y,z = self.body.get_position()
        self.pose.position.x = x
        self.pose.position.y = y
        self.pose.position.z = z

        x,y,z,w = self.body.get_orientation_quat()
        self.pose.orientation.x = x
        self.pose.orientation.y = y
        self.pose.orientation.z = z
        self.pose.orientation.w = w

        self.pose_stamped.header.stamp = rospy.Time.now()
        self.pose_stamped.pose = self.pose


    def update(self):
        self._update_pose()
        self.pub.publish(self.pose_stamped)


class RosMarkerArrayView:
    def __init__(self):
        self.last_used_id = -1

        self.pub = rospy.Publisher('/rviz_marker_array', MarkerArray, queue_size=1)
        self.marker_array = MarkerArray()

        self.ros_pose_viewers = []
        self.mesh_paths = []
        self.mesh_scales = []
        self.mesh_rgbas = []


    def add_view(self, pose_view, mesh_path, mesh_scale, mesh_rgba):
        self.ros_pose_viewers.append(pose_view)
        self.mesh_paths.append(mesh_path)
        self.mesh_scales.append(mesh_scale)
        self.mesh_rgbas.append(mesh_rgba)

    def init(self):
        for poseview,mesh_path,mesh_scale,mesh_rgba in zip(self.ros_pose_viewers,
                                                           self.mesh_paths,
                                                           self.mesh_scales,
                                                           self.mesh_rgbas):
            marker = Marker()
            marker.ns = '/marker_array'
            marker.id = self.last_used_id+1
            self.last_used_id += 1
            marker.action = 0
            # 10 for mesh
            marker.type = 10

            marker.pose = poseview.pose
            x,y,z = mesh_scale
            marker.scale.x = x
            marker.scale.y = y
            marker.scale.z = z

            r,g,b,a = mesh_rgba
            marker.color.a = a
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b

            marker.mesh_resource = 'file://'+mesh_path
            marker.header.frame_id = '/world'

            self.marker_array.markers.append(marker)

    def update(self):
        self.pub.publish(self.marker_array)


class RosSwarmView:
    def __init__(self,
                 swarm,
                 mesh_path,
                 mesh_rgba,
                 mesh_scale,
                 last_used_id=999):
        """
        Similar to marker array view, but this assumes that the given array of stuff
        are all identical.

        swarm needs to have get_positions() and get_orientation_quats() functions.

        if more than 1 swarm view is created, take care of last_used_id!
        """

        self.swarm = swarm
        #  self.last_used_id = -1
        self.last_used_id = last_used_id
        self.pub = rospy.Publisher('/rviz_marker_array', MarkerArray, queue_size=1)

        # create these ahead of time, just need to update before publishing
        self.marker_array = MarkerArray()

        self.poses = []
        for i in range(len(swarm._pos)):
            point = Point()
            quat = Quaternion()
            pose = Pose()
            pose.position = point
            pose.orientation = quat

            marker = Marker()
            marker.ns = '/marker_array'
            marker.id = self.last_used_id+1
            self.last_used_id += 1
            marker.action = 0
            # 10 for mesh
            marker.type = 10

            marker.pose = pose
            x,y,z = mesh_scale
            marker.scale.x = x
            marker.scale.y = y
            marker.scale.z = z

            r,g,b,a = mesh_rgba
            marker.color.a = a
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b

            marker.mesh_resource = 'file://'+mesh_path
            marker.header.frame_id = '/world'

            self.marker_array.markers.append(marker)



    def update(self):
        for pos,quat,marker in zip(self.swarm.get_position(),
                                   self.swarm.get_orientation_quat(),
                                   self.marker_array.markers):

            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]

            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

        self.pub.publish(self.marker_array)





if __name__=='__main__':
    rospy.init_node('rosviewtest', anonymous=True)

    N = 1

    ups = 30
    rate = rospy.Rate(ups)

    from dynamic_point import PIDPoint, VelocityPoint
    bodies = []
    pose_views = []
    for i in range(N):
        pidp = PIDPoint(damping=0.05)
        #  pidp = VelocityPoint(speed=1)
        rv = RosPoseView(body=pidp,
                         topic='/test'+str(i))
        bodies.append(pidp)
        pose_views.append(rv)

    marker_array = RosMarkerArrayView(ros_pose_viewers=pose_views,
                                      mesh_paths=['/home/ozer/suzanne.dae']*N,
                                      mesh_rgbas=[(0.8,0.2,0.5,1)]*N,
                                      mesh_scales=[(1,1,1)]*N)

    target_view = RosPoseView(body=bodies[0].target,
                              topic='/target0')

    i = 0
    import random

    bodies[0].set_target((1,1,0))

    while not rospy.is_shutdown():
        for body,poseview in zip(bodies,pose_views):
            if i%200 == 0:
                rx = -5+random.random()*10
                ry = -5+random.random()*10
                rz = -5+random.random()*10
                body.set_target((rx,ry,rz))
            body.update(1/ups)
            poseview.update()
            print('body:',body.pos, 't;', body.target.pos)

        marker_array.update()
        target_view.update()

        rate.sleep()
        i += 1
    else:
        print('rospy is_shutdown')






