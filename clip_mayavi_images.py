from pyntcloud import PyntCloud
from pyntcloud.io import bin as io_bin
import numpy as np
import pandas as pd
import os

from pythreejs import *
import numpy as np
from IPython.display import display
from ipywidgets import HTML, Text, Output, VBox
from traitlets import link, dlink
import glob
from tqdm import tqdm
from mayavi import mlab
import time 

VELODYNE_HEIGHT = 1.73

def extractBB(label,gt=True):
    label_split = label.split(" ")
    if gt:
        return label_split[0],float(label_split[-7]),float(label_split[-6]),float(label_split[-5]),float(label_split[-4]),float(label_split[-3]),float(label_split[-2]),float(label_split[-1])
    else:
        return label_split[0],float(label_split[-8]),float(label_split[-7]),float(label_split[-6]),float(label_split[-5]),float(label_split[-4]),float(label_split[-3]),float(label_split[-2])

def extractscore(label):
    label_split = label.split(" ")
    return label_split[0],float(label_split[-1])

def extractscoreonly(label):
    label_split = label.split(" ")
    print(label_split[-1])
    return float(label_split[-1])

def getCorners(height,width,length,x,y,z,θ,rotation=True):
    
    corners = np.array([[-length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2],
                        [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
                        [0, 0, 0, 0, height, height, height, height]])
    
    rotMat = np.array([[np.cos(θ) , -np.sin(θ) , 0],
                       [np.sin(θ) ,  np.cos(θ) , 0],
                       [    0     ,     0      , 1]])
    if rotation:
        cornersPos = (np.dot(rotMat,corners)+np.tile([x,y,z],(8,1)).T).transpose()
        corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = cornersPos[0],cornersPos[1],cornersPos[2],cornersPos[3],cornersPos[4],cornersPos[5],cornersPos[6],cornersPos[7]
    else:
        cornersPos = (corners + np.tile([x,y,z],(8,1)).T).transpose()
        corner1,corner2,corner3,corner4,corner5,corner6,corner7,corner8 = cornersPos[0],cornersPos[1],cornersPos[2],cornersPos[3],cornersPos[4],cornersPos[5],cornersPos[6],cornersPos[7]
    
    return list(corner1),list(corner2),list(corner3),list(corner4),list(corner5),list(corner6),list(corner7),list(corner8)

def createBBox(bounding_box,C1,C2,C3,C4,C5,C6,C7,C8,color="yellow"):
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C2,C3,C4,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C4,C8,C5,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C1,C2,C6,C5,C1]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C2,C6,C7,C3,C2]
        })
    bounding_box.append(
        {
            "color":color,
            "vertices":[C3,C7,C8,C4,C3]
        })
    return bounding_box

def draw_simple(pc,fig=None, engine = None):
#     if engine:
#         fig = mlab.figure(figure=figure, bgcolor=(0,0,0), fgcolor=None, engine=engine, size=(1600, 1000))
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
#     return fig

def draw_gt_boxes3d(gt_boxes3d,conf, thres = 0, color=(1,1,1), fig = None,engine = None, line_width=1, draw_text=True, text_scale=(0.5,0.5,0.5), color_list=None):
#     if engine:
#         fig = mlab.figure(figure=fig, engine =engine)
        
    num = len(gt_boxes3d)

    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if n >= thres:
            color = (0,1,0)
        else:
            color = (1,0,0)
        if draw_text and n >= thres and n-thres <= len(conf): mlab.text3d(b[3,0], b[3,1], b[3,2], '%.1f'%float(conf[n-thres]), scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,
                                                     1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show()
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    

filelist = glob.glob("/home/manojpc/PointRCNN/data/KITTI/object/training/velodyne/*")

numlist = list()
for each in range(len(filelist)):
    numlist.append(int(filelist[each][60:-4]))
sorted_list = sorted(numlist)    


time_result = list()

mlab.options.offscreen = True

for each in tqdm(sorted_list):
    PC_ID = '%06d' % each # [0 : 7480] 

    KITTI_PATH = "/home/manojpc/PointRCNN/data/KITTI/object/training/"
    PointRCNN_PATH2 = "/home/manojpc/PointRCNN/pretrained_model.pth"
    PointRCNN_PATH = "/home/manojpc/PointRCNN/resultdata/data/"

    points = io_bin.read_bin(KITTI_PATH+"velodyne/"+PC_ID+".bin",shape=(-1,4))['points']
    points['z'] += VELODYNE_HEIGHT

    if os.path.exists(KITTI_PATH+"label_2/"+PC_ID+".txt"):
        assert os.path.exists(KITTI_PATH+"label_2/"+PC_ID+".txt"), "File "+KITTI_PATH+"label_2/"+PC_ID+".txt doesn't exist !"
    else: continue
    
    file_label_gt = open(KITTI_PATH+"label_2/"+PC_ID+".txt","r")
    labels_gt = file_label_gt.readlines()
    file_label_gt.close()

    if os.path.exists(PointRCNN_PATH+PC_ID+".txt"):
        assert os.path.exists(PointRCNN_PATH+PC_ID+".txt"), "File "+PointRCNN_PATH+PC_ID+".txt doesn't exist !"
    else: continue
    
    file_label_pred = open(PointRCNN_PATH+PC_ID+".txt","r")
    labels_pred = file_label_pred.readlines()
    file_label_pred.close()

    labels_clean_gt = []
    for i,label in enumerate(labels_gt):
        labels_gt[i]=label[:-2]
        if labels_gt[i].split(" ")[0] in ["Car", "Van", "Truck"]:
            labels_clean_gt.append(extractBB(labels_gt[i],gt=True))

    labels_clean_pred = []
    for i,label in enumerate(labels_pred):
        labels_pred[i]=label[:-2]
        if labels_pred[i].split(" ")[0] == "Car":
            labels_clean_pred.append(extractBB(labels_pred[i],gt=False))

    bounding_box = []
    colors = {"GT":'red',"Pred":"blue"}

    boxes_gt = list()
    for label in labels_clean_gt: 
        object_type,height,width,length,x_tmp,y_tmp,z_tmp,θ = label
        x,y,z,θ = z_tmp,-x_tmp,y_tmp-VELODYNE_HEIGHT,np.pi/2-θ
        C1,C2,C3,C4,C5,C6,C7,C8 = getCorners(height,width,length,x,y,z,θ,rotation=True)
        bounding_box = createBBox(bounding_box,C1,C2,C3,C4,C5,C6,C7,C8,colors["GT"])
        boxes_gt.append(list((getCorners(height,width,length,x,y,z,θ,rotation=True))))

    boxes_pred = list()
    for label in labels_clean_pred: 
        object_type,height,width,length,x_tmp,y_tmp,z_tmp,θ = label
        x,y,z,θ = z_tmp,-x_tmp,y_tmp-VELODYNE_HEIGHT,np.pi/2-θ
        C1,C2,C3,C4,C5,C6,C7,C8 = getCorners(height,width,length,x,y,z,θ,rotation=True)
        bounding_box = createBBox(bounding_box,C1,C2,C3,C4,C5,C6,C7,C8,colors["Pred"])
        boxes_pred.append(list((getCorners(height,width,length,x,y,z,θ,rotation=True))))

    labels_clean_pred_score = []
    for i,label in enumerate(labels_pred):
        labels_pred[i]=label[:-2]
        if labels_pred[i].split(" ")[0] == "Car":
            labels_clean_pred_score.append(extractscore(labels_pred[i]))


    conf = []
    for each in labels_clean_pred_score:
        conf.append(each[1])
    conf = np.array(conf)

    if len(conf)==0 or np.max(conf) < 0.5:
    	maxconf = 1
    else:
    	maxconf = np.max(conf)

    if len(conf) == 0:
    	conf = np.array([0])
    	continue
    else:
    	conf = (conf-np.min(conf))/(maxconf-np.min(conf))

    all_box = list()
    #for each in range(len(bounding_box)):
    if len(boxes_gt):
        all_box.append(boxes_gt)
    if len(boxes_pred):
        all_box.append(boxes_pred)

    start = time.time()

    e = mlab.get_engine()

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))

    draw_simple(points.to_numpy().astype(float), fig=fig, engine = e)
    #fig = draw_lidar_colored(points.to_numpy().astype(float))

    fig.scene.parallel_projection = True

    thres = len(boxes_gt)
#     print(np.array(all_box).shape)
#     print(np.array(boxes_gt).shape)
#     print(np.array(boxes_pred).shape)
    all_boxes = np.concatenate(all_box)

    draw_gt_boxes3d(all_boxes, conf, thres = thres, color=(0,1,0), fig=fig, engine = e)

    #mlab.view(azimuth=0, elevation=45, focalpoint='auto', roll=90, figure=fig)
    #mlab.savefig("new6.png", figure=fig)
    scene = e.scenes[-1]
    cam = scene.scene.camera

    scene.scene.x_plus_view()
    cam.zoom(2.8)
    # scene.scene.save('/home/manojpc/snapshotx.png',size=(1600,1400))
    scene.scene.save('/home/manojpc/PointRCNN/imagelist/{}_1.png'.format(PC_ID), size=(1600,1400))
    # mlab.savefig("myfigurex.jpg", size=(1600,1400),figure=fig)
    scene.scene.y_plus_view()
    cam.zoom(1.9)
    scene.scene.save('/home/manojpc/PointRCNN/imagelist/{}_2.png'.format(PC_ID), size=(1600,1400))
    # mlab.savefig("myfigurey.jpg", size=(1600,1400),figure=fig)
    scene.scene.z_plus_view()
    cam.zoom(2.2)
    scene.scene.save('/home/manojpc/PointRCNN/imagelist/{}_3.png'.format(PC_ID), size=(1600,1400))
    # mlab.savefig("myfigurez.jpg", size=(1600,1400),figure=fig)
    # mlab.view(distance=20, figure=fig)

    mlab.clf(fig)
    mlab.close(all=True)
    # mlab.show()
    # print("closed at {}s".format(time.time()-start))
    time_result.append("" + str(PC_ID) + " " + str(time.time()-start))

np.savetxt("output.txt", np.array(time_result))

#draw_gt_boxes3d(np.asarray(boxes_pred), fig,color=(1,0,0))