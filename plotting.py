#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Author: Ozer Ozkahraman (ozkahramanozer@gmail.com)
# Date: 2018-08-03


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D #needed for '3d'
plt.ion()

try:
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
except ImportError:
    print('PyQtGraph of OpenGL not found! Can not use fancy graphing')


def make_opengl_fig(title='OpenGL Figure'):

    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle(title)
    g = gl.GLAxisItem()
    w.addItem(g)
    return w, app

def gl_line3(w, pts, c=None, size=None, color=None, cmap=None):
    pts = np.array(pts)
    if size is None:
        size = 1
    if color is None:
        color = np.ones((pts.shape[0], 4))

    if cmap is None:
        r,g,b = 0.3, 0.6, 1.0
    elif cmap=='random':
        r,g,b = np.random.rand(3)
    elif cmap=='red':
        r,g,b = 1, 0.3, 0.3
    elif cmap=='green':
        r,g,b = 0.3, 1, 0.3
    elif cmap=='blue':
        r,g,b = 0.3, 0.3, 1

    if c is not None:
        color = []
        # c is a 0-1 valued 'scale' for coloring per point
        for i in range(len(c)):
            color.append((r, g*c[i], b*c[i], 1))
        color = np.array(color)


    line = gl.GLLinePlotItem(pos=pts, width=size, color=color, antialias=False)
    w.addItem(line)
    return line

def gl_scatter3(w, pts, c=None, size=None, color=None):
    if size is None:
        size = np.ones((pts.shape[0]))
    if color is None:
        color = np.ones((pts.shape[0], 4))

    if c is not None:
        color = []
        # c is a 0-1 valued 'scale' for coloring per point
        for i in range(len(c)):
            color.append((0.3, 0.6*c[i], c[i], 0.8))
        color = np.array(color)


    scatter = gl.GLScatterPlotItem(pos=pts, size=size, color=color, pxMode=False)
    w.addItem(scatter)
    return scatter





def make_3d_fig():
    """
    create a figure and put a 3d axes in it.
    return the figure and axes objects
    """

    fig = plt.figure(figsize=(10,10))
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    ax.azim=45
    ax.elev=30
    #  plt.margins(tight=True)
    return fig, ax

def scatter3(ax, pts, **kwargs):
    """
    ax is a axes object from pyplot
    pts is a (N,3) numpy array
    """
    pts = np.atleast_2d(pts)
    return ax.scatter(pts[:,0], pts[:,1], pts[:,2], **kwargs)


def arrows3(ax, xyz, uvw, **kwargs):
    """
    ax is an axes object
    xyz are the starting points for arrows, (N,3)
    uvw are vectors, (N,3)
    """

    xyz = np.atleast_2d(xyz)
    uvw = np.atleast_2d(uvw)
    return ax.quiver(xyz[:,0], xyz[:,1], xyz[:,2],
                     uvw[:,0], uvw[:,1], uvw[:,2],
                     **kwargs)

def plot3(ax, pts, **kwargs):
    pts = np.atleast_2d(pts)
    return ax.plot3D(pts[:,0], pts[:,1], pts[:,2], **kwargs)


def surface_tri(ax, pts, uu, vv, colors=None, **kwargs):
    """
    plot weird stuff in 3D
    look at example:
    uu, vv = np.meshgrid([...], [...])
    colors = []
    for u in uu:
        for v in vv:
            pts.append( (x,y,z) )
            colors.append( value )
    #pts.shape == (N,3)
    """

    pts = np.atleast_2d(pts)
    # https://stackoverflow.com/questions/24218543/colouring-the-surface-of-a-sphere-with-a-set-of-scalar-values-in-matplotlib/24229480#comment37425657_24229480
    # these triangles are the indices of 1d-ified uu-vv's indices
    ruu, rvv = np.ravel(uu), np.ravel(vv)
    tri = mtri.Triangulation(ruu,rvv).triangles
    ret = ax.plot_trisurf(pts[:,0], pts[:,1], pts[:,2], triangles=tri, **kwargs)

    if colors is not None:
        colors = np.array(colors)
        # average the colors of the vertices for each triangle face
        # colors per triangle
        # colors[tri] basically chooses the vertices by triangles
        colors = np.max(colors[tri], axis=1)
        #  colors = np.mean(colors[tri], axis=1)
        ret.set_array(colors)
        ret.autoscale()

    return ret



