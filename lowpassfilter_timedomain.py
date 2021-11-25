import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import json



def lowpass_filter(reading, last_output, alpha):
    
    last_output = alpha * reading + (1 - alpha) * last_output

    return last_output


path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'

f=open(path_sample,encoding='utf-8') 
res=f.read() 
points = np.array(json.loads(res)['1']['cam_pc'])
visibility = np.array(json.loads(res)['1']['visibility'])

points_low = []
points_vis = []
last_point = np.array(points[:, 0])
for i in range(points.shape[1]):
    # print(points[:, i])
    # print(visibility[:, i])
    if visibility[i] == 1:
        point = np.array(points[:, i])
        point_low = lowpass_filter(point, last_point, 0.55)
        last_point = point_low
        print(point-point_low)
        points_low.append(point_low)
        points_vis.append(point)
        # print(points[:,i])
        # print(visibility[:,i])
points_low = np.array(points_low)        
points_vis = np.array(points_vis)

print(points.shape)
# print(visibility)
print(points_low.shape)
x1 = points_vis[:,0]
y1 = points_vis[:,1]
z1 = points_vis[:,2]
x = points_low[:,0]
y = points_low[:,1]
z = points_low[:,2]
# print(y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax1 = fig.add_subplot(projection='3d')
fig.add_subplot(projection='3d')
plt.plot(x1, y1, z1, label='ALPHA=0.1')
plt.plot(x, y, z)
ax.set(xlabel="X", ylabel="Y", zlabel="Z")

# ax1.legend()
plt.show()

# fig = plt.figure()
# # ax = fig.add_subplot(projection='2d')
# plt.plot(x, y, label='parametric curve')
# # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# # ax.legend()
# plt.show()



# if __name__ == '__main__':

#     print('PyCharm')
