#对于数据的初步处理，将视频文件转化为坐标储存

import os
import json
import cv2
import dlib
import numpy as np
def get_mouse_dlib(video_path,save_path,frame_detection = float("inf"),save_pic_or_not = None):
    """

    :param video_path: 视频路径
    :param save_path: 存储路径
    :param frame_detection: 处理多少帧，默认所有
    :param save_pic_or_not: 是否保存标记出68点图片
    :return: None
    """
    video_name = os.path.basename(video_path)[:-4]
    save_path = os.path.join(save_path, video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_picture = save_path + "/picture"
    save_mouse_picutre = save_path +"/mouse_picture"
    if not os.path.exists(save_picture):
        os.makedirs(save_picture)
    if not os.path.exists(save_mouse_picutre):
        os.makedirs(save_mouse_picutre)
    save_txt = save_path + "/txt"
    if not os.path.exists(save_txt):
        os.makedirs(save_txt)
        print("make_path")
    x_txt = open(save_txt+"/x.txt","a+") #记录刻画嘴部的20个点的横坐标，以及脸宽
    y_txt = open(save_txt+"/y.txt","+a") #记录刻画嘴部的20个点的纵坐标，以及脸宽

    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    zero = 0
    data_ = []
    all_data = {}
    if cap.isOpened():
        success = True

    else:
        success = False
        print("读取失败!")
        return None

    while (success and frame_detection > 0):
        frame_detection = frame_detection - 1
        success, frame = cap.read()
        if not success:
            break


        frame = np.rot90(frame, -1)
        
        frame = np.rot90(frame, -1)
        """ """
        detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        predictor = dlib.shape_predictor(r"C:\Users\admin\Desktop\net_for_mouse\project_for_mouse\shape_predictor_68_face_landmarks.dat")
        # 特征提取器的实例化
        dets = detector(frame, 1)
        print("人脸数：", len(dets))
        if len(dets)==0:
            print("未检测到人脸")
            cv2.imwrite(save_picture + "/{}wrong.jpg".format(zero), frame)
            zero += 1
            continue
        elif(len(dets)==1):
            frame_index += 1
            for k ,d in enumerate(dets):
                print("第", frame_index, "个人脸")
                width = d.right() - d.left()
                heigth = d.bottom() - d.top()
                shape = predictor(frame, d)
                print('人脸面积为：', (width * heigth))
                if save_pic_or_not:
                    # frame = frame[..., ::-1]
                    img = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                    img1 = img[int(max(shape.part(49).y, shape.part(50).y) - 5):int(
                        min(shape.part(57).y, shape.part(58).y) + 5),
                           int(shape.part(48).x - 5):int(shape.part(54).x + 5)]
                    cv2.imwrite(save_mouse_picutre+"/{}.jpg".format(frame_index),img1)
                    for i in range(68):
                        frame = cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), 1, 3)
                    cv2.imwrite(save_picture + "/{}.jpg".format(frame_index), frame)

                pose = []
                score = [1]*20
                skeleton = {}

                for j in range(48, 68):
                    x_txt.write(str(shape.part(j).x) + " ")
                    pose.append(shape.part(j).x)
                    pose.append(shape.part(j).y)
                x_txt.write(str(width))
                x_txt.write("\n")

                for j in range(48,68):
                    y_txt.write(str(shape.part(j).y) + " ")
                y_txt.write(str(heigth))
                y_txt.write("\n")

                skeleton["pose"] = pose
                skeleton["score"] = score
                skeleton_ = []
                skeleton_.append(skeleton)
                data = {}
                data["frame_index"] = frame_index
                data["skeleton"] = skeleton_
                data_.append(data)

            all_data["data"] = data_
            all_data["label"] = str(video_path.split("/")[-2])
            all_data["label_index"] = int(video_path.split("/")[-2])
            with open(save_path + "/"
                      + str(video_path.split("/")[-2])
                      + "-" + str(video_path.split("/")[-1][:-4])
                      + ".json", "w") as f:
                json.dump(all_data, f)

get_mouse_dlib(video_path = "D:/data_for_mouse/0/28.mov",save_path = "D:/data_for_mouse/0/",frame_detection = float("inf"),save_pic_or_not = True)
# get_mouse_dlib(video_path = "D:/data_for_mouse/0/28.mov",save_path = r"C:\Users\admin\Desktop",frame_detection = float("inf"),save_pic_or_not = True)


