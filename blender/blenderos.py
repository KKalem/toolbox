# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-09-06
#
# THIS FILE IS INTENDED TO BE USED FROM WITHIN BLENDER 2.76
# LOAD IT IN TO THE SAME NAMED .BLEND FILE AND RUN AS SCRIPT

import sys
import time
from queue import Queue

try:
    import bpy
except:
    print('YOU ARE NOT RUNNING THIS FROM WITHIN BLENDER, IMPORTING BPY FAILED')
    sys.exit(1)

try:
    import rospy
    from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
    from visualization_msgs.msg import Marker, MarkerArray
except:
    print('ROSPY NOT FOUND, IS BLENDER USING SYSTEM PYTHON OR BUNDLED PYTHON? IT SHOULD BE USING THE SYSTEM PYTHON OR YOU SHOULD SOMEHOW SHOW THE BUNDLED PYTHON WHERE ROSPY IS(good luck with that)')
    sys.exit(1)



class RosModalOperator(bpy.types.Operator):
    """
    Reads a queue of Markers or MarkerArrays and updates the objects in the blender scene
    with the data in them
    """
    bl_idname = 'wm.ros_modal_operator'
    bl_label = 'ROS Modal Operator'

    _timer = None

    def modal(self, context, event):
        if event.type in {'ESC'}:
            # ESC was pressed while running, stop running.
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            # we received a timed update notification from blender
            # this is when we update all the stuff from the queue
            global marker_q
            while marker_q.qsize() > 0:
                self.marker_update(marker_q.get(), context)

        # we did the thing, pass the context and such through to the rest of the stack of ops
        return {'PASS_THROUGH'}


    def execute(self, context):
        # first-time run, introduce ourselves to the window timer thingy and tell
        # it how frequent we want to run
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        global subscriptions
        [sub.unregister() for sub in subscriptions]
        print('Ended ROS connection')



    def marker_update(self, marker, context):
        """
        duplicate a base object, rename it, move it
        """

        # blender uses 'xxx.123' when duplicating stuff, so we will too
        idstr = str(marker.id)
        prepend = (3-len(idstr))*'0'
        object_name = marker.ns + '.' + prepend + idstr
        # we now know the name of our object
        # check if it exists, if not we will create it
        bl_obj = bpy.data.objects.get(object_name)
        if bl_obj is None:
            # object does not exist, we will dupe-linked an existing one
            # we expect to find an un-numbered object with the name ns
            bl_source_obj = bpy.data.objects.get(marker.ns)
            if bl_source_obj is None:
                # we have no object to move OR a source object to clone from
                print('No source object found to dupe from:',object_name)
                return

            # unselect everything so we dont dupe more than needed
            bpy.ops.object.select_all(action='DESELECT')
            # select the source object to be duplicated
            bl_source_obj.select = True
            # duplicate it
            res = bpy.ops.object.duplicate(linked=False)
            print('Duplicate run', bpy.data.objects.values())
            # the dupe is selected now

            if res != {'FINISHED'}:
                print('Could not duplicate source object:', bl_source_obj)
                return

            # after duping, the new object is selected by default
            # rename the object to what the marker wants it to be
            try:
                context.object.name = object_name
                print('Duped object:',object_name)
                # we can now select this new object by its name
                # this should now work!
                bl_obj = bpy.data.objects.get(object_name)
                # set rotation mode to quat
                bl_obj.rotation_mode = 'QUATERNION'
                # put it on the first layer
                bl_obj.layers = [False]*20
                bl_obj.layers[0] = True
                return
            except AttributeError:
                print('Context has no object while trying to give name:', object_name)
                return


        # now set the location and orientation of this
        pos = marker.pose.position
        bl_obj.location = (pos.x, pos.y, pos.z)
        # insert a keyframe if requested
        # seq is used as the frame number to insert the keyframe at
        if marker.frame_locked:
            bl_obj.keyframe_insert(data_path='location', frame=marker.header.seq)



        # dont touch orientation if theyre all 0s, maybe blender'll change it
        ori = marker.pose.orientation
        if not (ori.w == 0 and ori.x == 0 and ori.y == 0 and ori.z == 0):
            bl_obj.rotation_quaternion = (ori.w, ori.x, ori.y, ori.z)
            # insert a keyframe if requested
            # seq is used as the frame number to insert the keyframe at
            if marker.frame_locked:
                bl_obj.keyframe_insert(data_path='rotation_quaternion', frame=marker.header.seq)


        # scale it too
        bl_obj.scale = (marker.scale.x, marker.scale.y, marker.scale.z)
        # insert a keyframe if requested
        # seq is used as the frame number to insert the keyframe at
        if marker.frame_locked:
            bl_obj.keyframe_insert(data_path='scale', frame=marker.header.seq)


        col = marker.color
        if not (col.r == 0 and col.g == 0 and col.b == 0 and col.a == 0):
            # change the color of the material
            # this changes all colors that are duped from the same mesh object
            material_name = 'mat_'+object_name
            try:
                # does it have a specific material?
                bl_obj_mat = bl_obj.material_slots[material_name].material
            except KeyError:
                # this material does not exist in the object yet
                # does it exist in the scene?
                try:
                    # get it from the scene
                    bl_mat = bpy.data.materials[material_name]
                    print('Found material in scene:',material_name, bl_mat)
                except KeyError:
                    # it does not exit in the scene either, make it anew
                    bl_mat = bpy.data.materials.new(name=material_name)
                    print('Created material:', material_name, bl_mat)

                # set the first slot to be this mat
                bl_obj_mat = bl_mat
                if len(bl_obj.material_slots) > 0:
                    bl_obj.material_slots[0].material = bl_mat
                else:
                    bl_obj.material_slots.append(bl_mat)

            # change its color
            bl_obj_mat.diffuse_color[0] = col.r
            bl_obj_mat.diffuse_color[1] = col.g
            bl_obj_mat.diffuse_color[2] = col.b
            if marker.frame_locked:
                bl_obj_mat.keyframe_insert(data_path='diffuse_color', frame=marker.header.seq)

            # change the alpha
            if marker.color.a < 1:
                bl_obj_mat.use_transparency = True
                bl_obj_mat.transparency_method = 'Z_TRANSPARENCY'
                bl_obj_mat.alpha = marker.color.a
                if marker.frame_locked:
                    bl_obj_mat.keyframe_insert(data_path='alpha', frame=marker.header.seq)


        # add a keyframe
        if marker.frame_locked:
            frame = marker.header.seq





def register():
    bpy.utils.register_class(RosModalOperator)

def unregister():
    bpy.utils.unregister_class(RosModalOperator)

def marker_callback(marker):
    global marker_q
    if marker_q.qsize() < 50:
        marker_q.put(marker)
    return 1

def marker_array_callback(marker_array):
    [marker_callback(marker) for marker in marker_array.markers]


if __name__ == '__main__':
    print('======== ROS bridge running ========')
    register()

    # init ros node as usual
    bpy.ops.object.select_all(action='DESELECT')

    # this node needs to be UNIQUE
    rospy.init_node('blender', anonymous=False, disable_signals=True)

    # a global queue for the subs and modal op to use
    marker_q = Queue()

    # create a global list of subscribers so we can kill them later
    subscriptions = []
    # this creates a thread that will fill marker_q
    subscriptions.append(rospy.Subscriber('/rviz_swarm', MarkerArray, marker_array_callback))

    # run it
    bpy.ops.wm.ros_modal_operator()
