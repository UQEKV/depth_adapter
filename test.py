import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import json
import math
from scipy.fftpack import fft,ifft


# path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
# path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'
# def sort_points(p):
#
#     idx = np.argsort(p[:, 0])
#
#     return p[idx]
#
#
# f=open(path,encoding='utf-8')
# res=f.read()
# # points = np.array(json.loads(res)['1']['cam_pc'])
# print(list(json.loads(res).keys()))
#
# lane_id = list(json.loads(res).keys())
# for ID in lane_id:
#     print(ID)
#
# points = np.array(json.loads(res)['1']['cam_xyz'])
# visibility = np.array(json.loads(res)['1']['visibility'])
#
#
#
# points_vis = []
# sample_points = []
#
# for i in range(points.shape[1]):
#     if visibility[0][i] == 1:
#         point = np.array(points[:, i])
#         points_vis.append(point)
#
#
# points_vis = np.array(points_vis)
# points_vis = np.unique(points_vis,axis=0)
# points_vis = sort_points(points_vis)


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.fftpack import fft, ifft

a = np.mat([[1, 2, 3], [ 2, 3, 1]])
b = np.mat([[1], [2], [3]])
c = np.mat(np.eye(3))


print(c*b)











# class kalman_filter:
#     def __init__(self, Q, R,init):
#         self.Q = Q
#         self.R = R
#
#         self.P_k_k1 = 1
#         self.Kg = 0
#         self.P_k1_k1 = 1
#         self.x_k_k1 = 0
#         self.ADC_OLD_Value = init
#         self.Z_k = 0
#         self.kalman_adc_old = init
#
#     def kalman(self, ADC_Value):
#         self.Z_k = ADC_Value
#
#         # if (abs(self.kalman_adc_old-ADC_Value)>=60):
#         # self.x_k1_k1= ADC_Value*0.382 + self.kalman_adc_old*0.618
#         # else:
#         self.x_k1_k1 = self.kalman_adc_old;
#
#         self.x_k_k1 = self.x_k1_k1
#         self.P_k_k1 = self.P_k1_k1 + self.Q
#
#         self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)
#
#         kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
#         self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
#         self.P_k_k1 = self.P_k1_k1
#
#         self.kalman_adc_old = kalman_adc
#
#         return kalman_adc
#
#
# class RC_filter:
#     def __init__(self, sampleFrq, CutFrq, init):
#         self.sampleFrq = sampleFrq
#         self.CutFrq = CutFrq
#         self.adc_old = init
#
#     def LowPassFilter_RC_1order(self, Vi):
#         RC = 1.0 / 2.0 / math.pi / self.CutFrq
#         Cof1 = 1 / (1 + RC * self.sampleFrq)
#         Cof2 = RC * self.sampleFrq / (1 + RC * self.sampleFrq)
#         Vo = Cof1 * Vi + Cof2 * self.adc_old
#         self.adc_old = Vo
#         return Vo
#
#
# if __name__ == '__main__':
#
#     path_sample = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/sampled_json/150733833972875200.json'
#
#     f = open(path_sample, encoding='utf-8')
#     res = f.read()
#     points = np.array(json.loads(res)['2']['cam_pc'])
#     visibility = np.array(json.loads(res)['2']['visibility'])
#
#     points_low = []
#     points_vis = []
#     last_point = np.array(points[:, 0])
#     for i in range(points.shape[1]):
#         # print(points[:, i])
#         # print(visibility[:, i])
#         if visibility[i] == 1:
#             point = np.array(points[:, i])
#             # point_low = lowpass_filter_time(point, last_point, 0.1)
#             # last_point = point_low
#             # print(point - point_low)
#             # points_low.append(point_low)
#             points_vis.append(point)
#             # print(points[:,i])
#             # print(visibility[:,i])
#     # points_low = np.array(points_low)
#     points_vis = np.array(points_vis)
#     y = points_vis[:, 2]
#
#     noise_size = 1024
#     noise_size_half = 512
#     kalman_filter = kalman_filter(0.01, 0.4, y[0])
#     RC_filter = RC_filter(400, 5, y[0])
#     noise_array = np.random.normal(0, 2, noise_size)
#
#     adc_value = []
#
#     for i in range(noise_size):
#         adc_value.append(0)
#
#     adc_value_noise =  y       #np.array(adc_value) + noise_array
#     noise_size = len(y)
#     adc_filter_1 = []
#
#     for i in range(noise_size):
#         adc_filter_1.append(kalman_filter.kalman(adc_value_noise[i]))
#     plt.plot(adc_value_noise, 'r')
#     plt.plot(adc_filter_1, 'b')
#
#     # plt.plot(test_array)
#     plt.show()
#     adc_filter_2 = []
#     plt.figure(1)
#     for i in range(noise_size):
#         adc_filter_2.append(RC_filter.LowPassFilter_RC_1order(adc_value_noise[i]))
#
#     plt.plot(adc_value_noise, 'r-')
#     plt.plot(adc_filter_2, 'b')
#     # plt.plot(test_array)
#     plt.show()
#     plt.figure(2)
#
#     x = range(noise_size)
#     y = noise_array
#     # x=np.linspace(0,1,1400)
#     # 设置需要采样的信号，频率分量有180，390和600
#     # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
#
#     yy = fft(x)  # 快速傅里叶变换
#     yreal = yy.real  # 获取实数部分
#     yimag = yy.imag  # 获取虚数部分
#     yf = abs(fft(y))  # 取绝对值
#     yf1 = abs(fft(y)) / len(x)  # 归一化处理
#     yf2 = yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间
#     xf = np.arange(len(y))  # 频率
#     xf1 = xf
#     xf2 = xf[range(int(len(x) / 2))]  # 取一半区间
#     """plt.subplot(221)
#     plt.plot(x[0:noise_size_half],y[0:noise_size_half])
#     plt.title('Original wave')
#     plt.subplot(222)
#     plt.plot(xf,yf,'r')
#     plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B') #注意这里的颜色可以查询颜色代码表
#
#     plt.subplot(223)
#     plt.plot(xf1,yf1,'g')
#     plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r') """
#     # plt.subplot(224)
#     plt.plot(xf2, yf2, 'b')
#     plt.title('FFT of kalman_filter)', fontsize=10, color='#F08080')
#     plt.show()
#
#     x = range(noise_size)
#     y = adc_filter_2
#     # x=np.linspace(0,1,1400)
#     # 设置需要采样的信号，频率分量有180，390和600
#     # y=7*np.sin(2*np.pi*180*x) + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
#
#     yy = fft(x)  # 快速傅里叶变换
#     yreal = yy.real  # 获取实数部分
#     yimag = yy.imag  # 获取虚数部分
#     yf = abs(fft(y))  # 取绝对值
#     yf1 = abs(fft(y)) / len(x)  # 归一化处理
#     yf2 = yf1[range(int(len(x) / 2))]  # 由于对称性，只取一半区间
#     xf = np.arange(len(y))  # 频率
#     xf1 = xf
#     xf2 = xf[range(int(len(x) / 2))]  # 取一半区间
#     """plt.subplot(221)
#     plt.plot(x[0:noise_size_half],y[0:noise_size_half])
#     plt.title('Original wave')
#     plt.subplot(222)
#     plt.plot(xf,yf,'r')
#     plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B') #注意这里的颜色可以查询颜色代码表
#
#     plt.subplot(223)
#     plt.plot(xf1,yf1,'g')
#     plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r') """
#     # plt.subplot(224)
#     plt.plot(xf2, yf2, 'b')
#     plt.title('FFT of RC)', fontsize=10, color='#F08080')
#     plt.show()
#
#
