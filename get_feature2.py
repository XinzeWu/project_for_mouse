# 对于数据的进一步处理，将坐标文件进一步提取成特征文件，储存16个特征
import cv2
import os
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import fsolve


def caculate_kappa(x, y):
    """
    input  : the coordinate of the three point
    output : 输出三点组成的圆的半径
    """
    if (x[1]*y[2]-x[2]*y[1])+(x[2]*y[0]-x[0]*y[2])+(x[0]*y[1]-y[0]*x[1])==0:
        return float("inf")                           #三点围成的三角形面积为0

    def f1(xx):
        return np.array([(x[0] - xx[0]) ** 2 + (y[0] - xx[1]) ** 2 - xx[2]**2,
                         (x[1] - xx[0]) ** 2 + (y[1] - xx[1]) ** 2 - xx[2]**2,
                         (x[2] - xx[0]) ** 2 + (y[2] - xx[1]) ** 2 - xx[2]**2])

    sol_fsolve = fsolve(f1, [10 ,10, 10])
    print(sol_fsolve[2])
    print("误差",sum(f1(sol_fsolve)))
    return abs(sol_fsolve[2])


def caculate_area(coordinate):
    """
    该函数用来计算上下唇的面积
    :param coordinate: 总的坐标，要是三维的,直接输入numpy形式
    :return:返回两个值即上下唇的面积
    """
    index1 = [0,1,2,3,4,5,6,15,14,13]
    index2 = [19,18,17,16,7,8,9,10,11,12]
    coo1 = coordinate[index1]
    coo2 = coordinate[index2]

    return cv2.contourArea(coo1),cv2.contourArea(coo2)

def caculate_distance(coordinate):
    """
    用来计算上下唇之间的距离
    :param coordinate: 总的坐标，要是三维的,直接输入numpy形式
    :return: 返回距离
    """
    dis1 = dist.euclidean((coordinate[13][0][0], coordinate[13][0][1]), (coordinate[19][0][0], coordinate[19][0][1]))
    dis2 = dist.euclidean((coordinate[14][0][0], coordinate[14][0][1]), (coordinate[18][0][0], coordinate[18][0][1]))
    dis3 = dist.euclidean((coordinate[15][0][0], coordinate[15][0][1]), (coordinate[17][0][0], coordinate[17][0][1]))

    return (dis1+dis2+dis3)/3

def get_feature2(txt_path):
    datax,datay = [],[]
    for line in open(txt_path+"/x.txt"):
        datax.append(line)
    for line in open(txt_path+"/y.txt"):
        datay.append(line)
    lens = len(datax)

    if os.path.exists(txt_path + "/feature2.txt"):
        os.remove(txt_path + "/feature2.txt")
        print("delete")
    if os.path.exists(txt_path + "/kappa.txt"):
        os.remove(txt_path + "/kappa.txt")
        print("delete")

    feature2 = open(txt_path + "/feature2.txt", "a+")
    kappa = open(txt_path + "/kappa.txt", "a+")
    for i in range(lens):
        data = []                #这里的data应该是三维的坐标
        for j in range(20):      # 从TXT文件中获得的是20个坐标以及人脸长和宽
            data.append([[int(datax[i].split()[j]),int(datay[i].split()[j])]])
        data = np.array(data)
        dis = caculate_distance(data)
        face_wide = int(datax[i].split()[-1])
        face_high = int(datay[i].split()[-1])
        mouse_wide = dist.euclidean((data[0][0][0], data[0][0][1]), (data[6][0][0], data[6][0][1]))
        area1, area2 = caculate_area(data)     #上下唇的面积

        normal_dis = dis/face_high             #嘴巴张开的距离（利用脸长标准化）
        normal_wide = mouse_wide/face_wide     #嘴巴的宽度（利用脸宽标准化)
        normal_area1 = area1/mouse_wide
        normal_area2 = area2/mouse_wide
        feature2.write(str(normal_dis)+ " ")
        feature2.write(str(normal_wide) + " ")
        feature2.write(str(normal_area1) + " ")
        feature2.write(str(normal_area2) + "\n")
        # 12个三点圆的半径
        ka0 = caculate_kappa(data[0:3, 0, 0], data[0:3, 0, 1])
        ka1 = caculate_kappa(data[1:4, 0, 0], data[1:4, 0, 1])
        ka2 = caculate_kappa(data[2:5, 0, 0], data[2:5, 0, 1])
        ka3 = caculate_kappa(data[3:6, 0, 0], data[3:6, 0, 1])
        ka4 = caculate_kappa(data[4:7, 0, 0], data[4:7, 0, 1])
        ka5 = caculate_kappa(data[13:16, 0, 0], data[13:16, 0, 1])
        ka6 = caculate_kappa(data[17:20, 0, 0], data[17:20, 0, 1])
        ka7 = caculate_kappa(data[10:13, 0, 0], data[10:13, 0, 1])
        ka8 = caculate_kappa(data[9:12, 0, 0], data[9:12, 0, 1])
        ka9 = caculate_kappa(data[8:11, 0, 0], data[8:11, 0, 1])
        ka10 = caculate_kappa(data[7:10, 0, 0], data[7:10, 0, 1])
        ka11= caculate_kappa(data[[8,7,16], 0, 0], data[[8,7,16], 0, 1])

        kappa.write(str(ka0) + " ")
        kappa.write(str(ka1) + " ")
        kappa.write(str(ka2) + " ")
        kappa.write(str(ka3) + " ")
        kappa.write(str(ka4) + " ")
        kappa.write(str(ka5) + " ")
        kappa.write(str(ka6) + " ")
        kappa.write(str(ka7) + " ")
        kappa.write(str(ka8) + " ")
        kappa.write(str(ka9) + " ")
        kappa.write(str(ka10) + " ")
        kappa.write(str(ka11) + "\n")


for i in range(31):
    get_feature2(r"D:\data_for_mouse\0\{}\txt".format(i))
print("-"*15,0,"-"*15)
for i in range(9):
    get_feature2(r"D:\data_for_mouse\1\{}\txt".format(i))
print("-"*15,1,"-"*15)
for i in range(20):
    get_feature2(r"D:\data_for_mouse\2\{}\txt".format(i))
print("-"*15,2,"-"*15)
for i in range(19):
    get_feature2(r"D:\data_for_mouse\3\{}\txt".format(i))
print("-"*15,3,"-"*15)
for i in range(13):
    get_feature2(r"D:\data_for_mouse\4\{}\txt".format(i))
print("-"*15,4,"-"*15)

