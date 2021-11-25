import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json


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



def sort_points(p):
    
    idx = np.argsort(p[:, 0])

    return p[idx]




def segments_fit_2d(X, Y, number_segments):
    xmin = X.min()
    xmax = X.max()
    print(xmax - xmin)
    seg = np.full(number_segments - 1, (xmax - xmin) / number_segments)
    print(seg.shape, (xmax - xmin) / number_segments)
    print(np.r_[np.r_[xmin, seg].cumsum(), xmax])
    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init]) # 0.01 is a threshold to constrain the searching interval
    def func(p):
        seg = p[:number_segments - 1]
        py = p[number_segments - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        # print(px.shape, py.shape)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)




def segments_fit_3d(X, Y, Z, number_segments):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(number_segments - 1, (xmax - xmin) / number_segments)
    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])
    pz_init = np.array([Z[np.abs(X-x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def segment_point(p):
        seg = p[:number_segments - 1]
        py = p[number_segments - 1:2*number_segments]
        pz = p[2*number_segments:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py, pz

    def mean_err(p):
        px, py, pz = segment_point(p)
        Y2 = np.interp(X, px, py)
        Z2 = np.interp(X, px, pz)
        return np.mean((Y - Y2)**2 + (Z - Z2)**2)
    r = optimize.minimize(mean_err, x0=np.r_[seg, py_init, pz_init], method='Nelder-Mead')
    return segment_point(r.x)



def load_json(path):
    f = open(path,encoding='utf-8') 
    res = f.read()   
    data = json.loads(res)
    return data


def segment_GO(data, pc_density, number):

    # points = np.array(data['1']['cam_pc'])
    lane_id = list(data.keys())

    lanes_vis = {}
    lanes_fit = {}
    for ID in lane_id:
        points_vis = []
        sample_points = []
        points = np.array(data[ID][pc_density])
        visibility = np.array(data[ID]['visibility'])

        for i in range(points.shape[1]):
            if visibility[0][i] == 1:
                point = np.array(points[:, i])
                points_vis.append(point)


        points_vis = np.array(points_vis)  
        points_vis = np.unique(points_vis,axis=0) 
        points_vis = sort_points(points_vis)  
        X = points_vis[:,0]
        Y = points_vis[:, 1]
        Z = points_vis[:, 2]
        px, py, pz = segments_fit_3d(X, Y, Z, number)
        lanes_fit[ID] = np.c_[px, py, pz]
        lanes_vis[ID] = np.c_[X, Y, Z]
        
    
    return lanes_fit, lanes_vis



        # plt.plot(X, Y)
        # # plt.plot(px, py, "-or");
        # plt.show()



path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'

data = load_json(path)
pc_density = 'cam_xyz'

lanes_fit, lanes_vis = segment_GO(data, pc_density, 17)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
colorlist=plt.cm.cool( np.linspace(0,1,len( list( lanes_fit.keys() ) ) ) )
for i, ID in enumerate(lanes_fit.keys()):
    lane_fit = lanes_fit[ID]
    lane_vis = lanes_vis[ID]
    px = lane_fit[:,0]
    py = lane_fit[:, 1]
    pz = lane_fit[:, 2]
    # X = lane_vis[:,0]
    # Y = lane_vis[:, 1]
    # Z = lane_vis[:, 2]


# fig.add_subplot(projection='3d')
    # plt.plot(X, Y, Z, ".")
    plt.plot(px, py, pz,"-or", color=colorlist[i]);
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
ax.legend()
plt.show()