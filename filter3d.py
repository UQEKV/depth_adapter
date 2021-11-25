import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import json

def load_json(path):
    f = open(path,encoding='utf-8')
    res = f.read()
    data = json.loads(res)
    return data

def sort_points(p):
    idx = np.argsort(p[:, 0])

    return p[idx]

def dataprecessing(path, pc_density):

    data = load_json(path)
    lane_id = list(data.keys())
    lanes_vis = {}
    for ID in lane_id:
        if ID.isdigit():
            points_vis = []
            points = np.array(data[ID][pc_density])
            visibility = np.array(data[ID]['visibility'])

            for i in range(points.shape[1]):
                if pc_density == 'cam_pc':
                    if visibility[i] == 1:
                        point = np.array(points[:, i])
                        points_vis.append(point)
                elif pc_density == 'cam_xyz':
                    if visibility[0][i] == 1:
                        point = np.array(points[:, i])
                        points_vis.append(point)

            points_vis = np.array(points_vis)
            points_vis = np.unique(points_vis, axis=0)
            points_vis = sort_points(points_vis)
            lanes_vis[ID] = points_vis
    return lanes_vis


def show3d(lanes_noise, lanes_kf):
    x = lanes_noise[:, 0]
    y = lanes_noise[:, 1]
    z = lanes_noise[:, 2]

    x1 = lanes_kf[:, 0]
    y1 = lanes_kf[:, 1]
    z1 = lanes_kf[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.plot(x, y, z, 'r')
    plt.plot(x1, y1, z1, 'b')
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.legend()
    plt.show()

def show3d_all(lanes_vis, lanes_f):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colorlist = plt.cm.cool(np.linspace(0, 1, len(list(lanes_vis.keys()))))
    for i, ID in enumerate(lanes_vis.keys()):
        lane_vis = lanes_vis[ID]
        lane_f = lanes_f[ID]
        # x = lane_vis[:, 0]
        # y = lane_vis[:, 1]
        # z = lane_vis[:, 2]
        x1 = lane_f[:, 0]
        y1 = lane_f[:, 1]
        z1 = lane_f[:, 2]
        # fig.add_subplot(projection='3d')
        # plt.plot(x, y, z)
        plt.plot(x1, y1, z1, color=colorlist[i]);
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.legend()
    plt.show()


def show2d(lanes_noise, lanes_kf, dim):
    plt.plot(lanes_noise[:, dim], 'r')
    plt.plot(lanes_kf[:, dim], 'b')
    plt.show()

class lowpass_filter_t:
    def __init__(self):
        self.alpha = 0.1
        self.x_last = np.mat([[0], [0], [0]])
        self.x_now = np.mat([[0], [0], [0]])

    def lowpass(self, value):
        self.x_now = value
        x_update = self.alpha * self.x_now + (1 - self.alpha) * self.x_last
        self.x_last = x_update
        return x_update



class kalman_filter:
    def __init__(self):
        self.Q = np.mat(np.diag([0.008, 0.008, 0.0064])) #x:0.008, 0.01  y:0.008, 0.01    z: 0.016, 0.4
        self.R = np.mat(np.diag([0.01, 0.01, 0.49]))
        self.F = np.mat(np.eye(3))
        self.H = np.mat(np.eye(3))
        self.dim = 3
        self.P_k_k1 = np.mat(np.eye(self.dim))
        self.Kg = np.mat(np.zeros((self.dim,self.dim)))
        self.P_k1_k1 = np.mat(np.eye(self.dim))
        self.x_k_k1 = np.mat(np.zeros((self.dim,1)))
        self.Z_k = np.mat(np.zeros((self.dim,self.dim)))
        self.kalman_old = np.mat([[0], [0], [0]])

    def kalman(self, Value):
        self.Z_k = Value
        self.x_k1_k1 = self.F * self.kalman_old;
        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.F * self.P_k1_k1 * self.F.T + self.Q
        self.Kg = self.P_k_k1 * self.H.T * (self.H * self.P_k_k1 * self.H.T + self.R).I
        kalman = self.x_k_k1 + self.Kg * (self.Z_k - self.H * self.kalman_old)
        self.P_k1_k1 = (np.mat(np.eye(self.dim)) - self.Kg * self.H) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1
        self.kalman_old = kalman

        return kalman


if __name__ == '__main__':

    path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
    path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'
    pc_density = 'cam_pc'

    lanes_vis = dataprecessing(path_sample, pc_density)
    # y = lanes_vis['6'][:, 2]
    # y1 = signal.medfilt(y,9)
    # print(y1)
    # plt.plot(y)
    # plt.plot(y1)
    # plt.show()


    print(lanes_vis.keys())
    kalman_filter = kalman_filter()
    lowpass_filter_t = lowpass_filter_t()
    lanes_kf = {}
    lanes_low = {}
    for ID in list(lanes_vis.keys()):
    # ID = '6'
        y = lanes_vis[ID]
        init = np.mat(y[0, :]).T
        noise_size = len(y)
        print(noise_size)
        kalman_filter.kalman_old = init
        lowpass_filter_t.x_last = init
        lanes_noise =  y
        lane_kf = []
        lane_low = []
        for i in range(noise_size):
            value = np.mat(lanes_noise[i]).T
            lane_kf.append(kalman_filter.kalman(value))
            lane_low.append(lowpass_filter_t.lowpass(value))
        lane_kf = np.array(lane_kf)
        lane_kf = lane_kf.reshape(noise_size,3)
        lanes_kf[ID] = lane_kf

        lane_low = np.array(lane_low)
        lane_low = lane_low.reshape(noise_size, 3)
        lanes_low[ID] = lane_low

    show3d_all(lanes_vis, lanes_kf)
    show3d_all(lanes_vis, lanes_low)
        # show3d(lanes_noise, lanes_kf[ID])
        # show2d(lanes_noise, lanes_kf[ID],2)
