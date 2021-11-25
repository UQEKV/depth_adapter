import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, ifft
import json

class kalman_filter:
    def __init__(self, Q, R,init):
        self.Q = Q
        self.R = R

        self.P_k_k1 = 1
        self.Kg = 0
        self.P_k1_k1 = 1
        self.x_k_k1 = 0
        self.ADC_OLD_Value = init
        self.Z_k = 0
        self.kalman_adc_old = init

    def kalman(self, ADC_Value):
        self.Z_k = ADC_Value

        # if (abs(self.kalman_adc_old-ADC_Value)>=60):
        # self.x_k1_k1= ADC_Value*0.382 + self.kalman_adc_old*0.618
        # else:
        self.x_k1_k1 = self.kalman_adc_old;

        self.x_k_k1 = self.x_k1_k1
        self.P_k_k1 = self.P_k1_k1 + self.Q

        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.kalman_adc_old = kalman_adc

        return kalman_adc




if __name__ == '__main__':

    path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'

    f = open(path_sample, encoding='utf-8')
    res = f.read()
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
            # point_low = lowpass_filter_time(point, last_point, 0.1)
            # last_point = point_low
            # print(point - point_low)
            # points_low.append(point_low)
            points_vis.append(point)
            # print(points[:,i])
            # print(visibility[:,i])
    # points_low = np.array(points_low)
    points_vis = np.array(points_vis)
    y = points_vis[:, 2]
    noise_size = len(y)
    noise_size_half = int(noise_size/2)

    Q = 0.016
    R = 0.4
    kalman_filter = kalman_filter(Q, R, y[0])   #x:0.008, 0.01  y:0.008, 0.01    z: 0.01, 0.4

    adc_value_noise =  y       #np.array(adc_value) + noise_array
    noise_size = len(y)
    adc_filter_1 = []

    for i in range(noise_size):
            adc_filter_1.append(kalman_filter.kalman(adc_value_noise[i]))
    adc_filter_1 = np.array(adc_filter_1)

    x = points_vis[:,0]
    y = points_vis[:,1]
    z_filtered = adc_filter_1
    z = points_vis[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # print(adc_filter_1)
    plt.plot(x, y, z, 'r')
    plt.plot(x, y, z_filtered, 'b')
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.legend()
    plt.show()

    plt.plot(adc_value_noise, 'r')
    plt.plot(adc_filter_1, 'b')

    plt.show()
