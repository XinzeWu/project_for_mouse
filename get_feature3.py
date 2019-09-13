# 对于16个特征进行处理，直接得到可以运用到svm的特征进行尝试
#12个弧度的半径 嘴巴张开大小 嘴巴宽度 上唇面积 下唇面积  分别标号为0~15
import numpy as np
import csv
def get_feature3(txt_path,save_csv = r"D:\data_for_mouse\mouse_200.csv",is_train = "true"):
    feature2,kappa = [],[]
    for line in open(txt_path+"/feature2.txt"):
        temp = line.split()
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        feature2.append(temp)


    for line in open(txt_path+"/kappa.txt"):
        temp = line.split()
        for i in range(len(temp)):
            temp[i] = float(temp[i])
        kappa.append(temp)

    feature2 , kappa = np.array(feature2) , np.array(kappa)
    _feature2 , _kappa = feature2.T , kappa.T                     #转置之后每一行是一个特征的所有帧的数据

    for i in range(12):
        where_are_inf = np.isinf(_kappa[i])
        _kappa[i][where_are_inf] = np.median(_kappa[i])  # 将inf转化成这里面中位数
    for i in range(4):
        where_are_inf = np.isinf(_feature2[i])
        _feature2[i][where_are_inf] = np.median(_feature2[i])

    flame = len(_kappa[0])
    print(flame)
    n = flame // 200
    data = []
    for j in range(n):
        line = []
        if is_train:
            line.append(txt_path[18])
        for i in range(12):
            k = _kappa[i][j*200:(j+1)*200]
            line.append(str(np.median(k)))
            line.append(str(np.mean  (k)))
            line.append(str(np.var   (k)))
            line.append(str(np.max   (k)))
            line.append(str(np.min   (k)))
        for i in range(4):
            f = _feature2[i][j*200:(j+1)*200]
            line.append(str(np.median(f)))
            line.append(str(np.mean  (f)))
            line.append(str(np.var   (f)))
            line.append(str(np.max   (f)))
            line.append(str(np.min   (f)))
        if sum(np.isnan(line)) + sum(np.isinf(line)) > 0:   # 如果改行中含有nan和inf，则不保存
            continue
        else:
            data.append(line)
    # 写入多行用writerows
    print(np.shape(data))

    with open(save_csv, "a+",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

with open(r"D:\data_for_mouse\mouse_200.csv", "a+",newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 先写入columns_name
    writer.writerow(["label",
                     "0_median", "0_med", "0_var", "0_max", "0_min",
                     "1_median", "1_med","1_var","1_max","1_min",
                     "2_median","2_med","2_var","2_max","2_min",
                     "3_median","3_med","3_var","3_max","3_min",
                     "4_median","4_med","4_var","4_max","4_min",
                     "5_median","5_med","5_var","5_max","5_min",
                     "6_median","6_med","6_var","6_max","6_min",
                     "7_median","7_med","7_var","7_max","7_min",
                     "8_median","8_med","8_var","8_max","8_min",
                     "9_median","9_med","9_var","9_max","9_min",
                     "10_median","10_med","10_var","10_max","10_min",
                     "11_median","11_med","11_var","11_max","11_min",
                     "12_median","12_med","12_var","12_max","12_min",
                     "13_median","13_med","13_var","13_max","13_min",
                     "14_median","14_med","14_var","14_max","14_min",
                     "15_median","15_med","15_var","15_max","15_min"])

for i in range(31):
    get_feature3(r"D:\data_for_mouse\0\{}\txt".format(i))
print("-"*15,0,"-"*15)
for i in range(9):
    get_feature3(r"D:\data_for_mouse\1\{}\txt".format(i))
print("-"*15,1,"-"*15)
for i in range(20):
    get_feature3(r"D:\data_for_mouse\2\{}\txt".format(i))
print("-"*15,2,"-"*15)
for i in range(19):
    get_feature3(r"D:\data_for_mouse\3\{}\txt".format(i))
print("-"*15,3,"-"*15)
for i in range(13):
    get_feature3(r"D:\data_for_mouse\4\{}\txt".format(i))
print("-"*15,4,"-"*15)




