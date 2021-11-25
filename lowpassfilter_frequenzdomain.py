import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import json
from scipy.fftpack import fft, ifft
import matplotlib.pylab

def lowpass_filter_time(reading, last_output, alpha):
    last_output = alpha * reading + (1 - alpha) * last_output

    return last_output


def sort_points(p):
    idx = np.argsort(p[:, 0])

    return p[idx]

# def lowpass_filter_frquenz(data,


path = '/home/PJLAB/chenjiangqiu/Downloads/validation/segment-14931160836268555821_5778_870_5798_870_with_camera_labels/lane_points/150733833972875200.json'
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
points_vis = sort_points(points_vis)

y = points_vis[:, 2]
# print(y)
# plt.plot(y)
# plt.show()
yy=fft(y)                     #快速傅里叶变换
yf=abs(fft(y))                # 取模
yf1=abs(fft(y))/((len(y)/2))           #归一化处理
print(abs(fft(y))/((len(y)/2)) )
yf2 = yf1[range(int(len(y)/2))]  #由于对称性，只取一半区间
#混合波的FFT（双边频率范围）
xf = np.arange(len(yy))
plt.figure(1)
plt.plot(xf,abs(fft(y)) ,'r') #显示原始信号的FFT模值
plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')
plt.show()
#
yy=fft(y)                     #快速傅里叶变换
yreal = yy.real               # 获取实数部分
yimag = yy.imag               # 获取虚数部分
test_y =yy
for i in range(len(yy)):
    if i>=50 and i<=250:
        test_y[i]=0
test = np.fft.ifft(test_y)  #对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
# print(abs(test))
y=test
yy=fft(y)                     #快速傅里叶变换
yf=abs(fft(y))                # 取模
yf1=abs(fft(y))/((len(y)/2))           #归一化处理
yf2 = yf1[range(int(len(y)/2))]  #由于对称性，只取一半区间
#混合波的FFT（双边频率范围）
xf = np.arange(len(y))
# plt.figure(2)
# plt.plot(xf,yf,'r') #显示原始信号的FFT模值
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表
# plt.show()
plt.plot(points_vis[:,2])
plt.plot(np.fft.ifft(test_y))
plt.show()



# print(points.shape)
# # print(visibility)
# print(points_low.shape)
# x1 = points_vis[:, 0]
# y1 = points_vis[:, 1]
# z1 = points_vis[:, 2]
# x = points_low[:, 0]
# y = points_low[:, 1]
# z = points_low[:, 2]
# # print(y)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax1 = fig.add_subplot(projection='3d')
# fig.add_subplot(projection='3d')
# plt.plot(x1, y1, z1, label='ALPHA=0.1')
# plt.plot(x, y, z)
# ax.set(xlabel="X", ylabel="Y", zlabel="Z")
#
# # ax1.legend()
# plt.show()

# fig = plt.figure()
# # ax = fig.add_subplot(projection='2d')
# plt.plot(x, y, label='parametric curve')
# # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# # ax.legend()
# plt.show()


# if __name__ == '__main__':

#     print('PyCharm')
# noise_size = 600
# noise_array = np.random.normal(0, 2, noise_size)
#
# adc_value = []
#
# for i in range(noise_size):
#     adc_value.append(0)
#
# y = np.array(adc_value) + noise_array
# print(y)
# plt.plot(np.array(y))
# plt.show()
#
# yy = fft(y)  # 快速傅里叶变换
# yf = abs(fft(y))  # 取模
# yf1 = abs(fft(y)) / ((len(y) / 2))  # 归一化处理
# yf2 = yf1[range(int(len(y) / 2))]  # 由于对称性，只取一半区间
# # 混合波的FFT（双边频率范围）
# xf = np.arange(len(y))
# plt.figure(1)
# plt.plot(xf, yf1, 'r')  # 显示原始信号的FFT模值
# plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')  # 注意这里的颜色可以查询颜色代码表
# plt.show()