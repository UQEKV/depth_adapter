import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import json


def sort_points(p):
    
    idx = np.argsort(p[:, 0])

    return p[idx]



def distance_3d(p1, p2):
    
    return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )**(1/3)
    # return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(1/2)

def linear_fitting_3D_points(points):
    # x = k1 * z + b1
    # y = k2 * z + b2
 
    Sum_X=0.0
    Sum_Y=0.0
    Sum_Z=0.0
    Sum_XZ=0.0
    Sum_YZ=0.0
    Sum_Z2=0.0
 
    for i in range(0,len(points)):
        xi=points[i][0]
        yi=points[i][1]
        zi=points[i][2]
 
        Sum_X = Sum_X + xi
        Sum_Y = Sum_Y + yi
        Sum_Z = Sum_Z + zi
        Sum_XZ = Sum_XZ + xi*zi
        Sum_YZ = Sum_YZ + yi*zi
        Sum_Z2 = Sum_Z2 + zi**2
 
    n = len(points)
    den = n*Sum_Z2 - Sum_Z * Sum_Z 
    k1 = (n*Sum_XZ - Sum_X * Sum_Z)/ den
    b1 = (Sum_X - k1 * Sum_Z)/n
    k2 = (n*Sum_YZ - Sum_Y * Sum_Z)/ den
    b2 = (Sum_Y - k2 * Sum_Z)/n
    
    para = [k1, b1, k2, b2]
    return k1, b1, k2, b2


def linear_3d(para, x):
 
    k1 = para[0]
    k2 = para[2]
    b1 = para[1]
    b2 = para[3]

    # x = k1 * z + b1
    # y = k2 * z + b2
    z = (x-b1)/k1
    y = k2 * z + b2

    return [x, y, z]




path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'

f=open(path,encoding='utf-8') 
res=f.read() 
# points = np.array(json.loads(res)['1']['cam_pc'])
points = np.array(json.loads(res)['1']['cam_xyz'])
visibility = np.array(json.loads(res)['1']['visibility'])



points_vis = []
sample_points = []

for i in range(points.shape[1]):
    if visibility[0][i] == 1:
        point = np.array(points[:, i])
        points_vis.append(point)

        if len(points_vis)==1:
            
                sample_points.append(point)

        # p1 = sample_points[-1]   
        # p2 = point
        # d =  distance_3d(p1,p2)
        # print(d)
        # print(p1)
        # print(p2)
        # if d > 1:
        #     sample_points.append(point)


points_vis = np.array(points_vis)  
points_vis = np.unique(points_vis,axis=0) 
points_vis = sort_points(points_vis)  

deltas_z = []
last_z = points_vis[0,2]
for i in range(points_vis.shape[0]):
    z_now = points_vis[i, 2]
    delta_z = z_now - last_z
    last_z = z_now
    deltas_z.append(delta_z)


# fig = plt.figure()
# # my_x_ticks = np.arange(1, 14, 1)
# # plt.xticks(my_x_ticks)
# # ax = fig.add_subplot(projection='2d')
# plt.plot(points_vis[:,0], deltas_z, label='delta z')
# # plt.grid()
# # plt.gca().set_aspect(1)
# plt.xlabel("X(m)")
# plt.ylabel("Delta Z(m)")

# # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# # ax.legend()
# plt.show()



last_sample_point = sample_points[-1] 
middle_points = []
for point in points_vis:
    p1 = sample_points[-1]   
    p2 = point
    d =  distance_3d(p1,p2)
    # print(d)
    # print(p1)
    # print(p2)
    if d > 4:
        sample_point_now = point
        sample_points.append(sample_point_now)
        middle_points.append((sample_point_now + last_sample_point)/2)
        last_sample_point = p1


sample_points = np.array(sample_points)
sample_points = sort_points(sample_points)   
middle_points = np.array(middle_points)
middle_points = sort_points(middle_points)   
print(points.shape)
print(points_vis.shape)
print(sample_points.shape)

x1 = points_vis[:,0]
y1 = points_vis[:,1]
z1 = points_vis[:,2]

x2 = sample_points[:,0]
y2 = sample_points[:,1]
z2 = sample_points[:,2]

x = middle_points[:,0]
y = middle_points[:,1]
z = middle_points[:,2]

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(projection='3d')
# ax.plot(x, y, z, label='parametric curve')
# ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# ax.legend()
# plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax1 = fig.add_subplot(projection='3d')
# fig.add_subplot(projection='3d')
# plt.plot(x1, y1, z1, label='ALPHA=0.1') #blue
# plt.plot(x2, y2, z2)  #orange
# plt.plot(x, y, z)
# ax.set(xlabel="X", ylabel="Y", zlabel="Z")

# # ax1.legend()
# plt.show()


fig = plt.figure()
# ax = fig.add_subplot(projection='2d')
plt.plot(x1, z1, label='parametric curve')
# ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# ax.legend()
plt.show()