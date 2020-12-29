import os
import sys
import random
import itertools
import colorsys
import numpy as np

from skimage.measure import find_contours
from PIL import Image
import cv2
import random
import math
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)

#---------------------------------------------------------#
#  Visualization
#---------------------------------------------------------#
def random_colors(N, bright=True):
    """
    生成随机颜色
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors

#计算两点的距离
def calDis(seq):
    dis=math.sqrt((seq[0][0]-seq[1][0])**2+(seq[0][1]-seq[1][1])**2)
    return dis

#暴力算法主体函数
def calDirect(seq):
    maxDis = 0.
    pair=[]
    for i in range(len(seq)):
        for j in range(i+1,len(seq)):
            dis = calDis([seq[i],seq[j]])
            if dis > maxDis:
                # print(dis)
                maxDis = dis
                pair=[seq[i],seq[j]]

    return [pair, maxDis]


# 生成器：生成横跨跨两个点集的候选点
def candidateDot(u, right, dis, med_x):
    cnt = 0
    # 遍历right（已按横坐标升序排序）。若横坐标小于med_x-dis则进入下一次循环；若横坐标大于med_x+dis则跳出循环；若点的纵坐标好是否落在在[u[1]-dis,u[1]+dis]，则返回这个点
    for v in right:
        if v[0] < med_x - dis:
            continue
        if v[0] > med_x + dis:
            break
        if v[1] >= u[1] - dis and v[1] <= u[1] + dis:
            yield v


# 求出横跨两个部分的点的最小距离
def combine(left, right, resMin, med_x):
    dis = resMin[1]
    minDis = resMin[1]
    pair = resMin[0]
    for u in left:
        if u[0] < med_x - dis:
            continue
        for v in candidateDot(u, right, dis, med_x):
            dis = calDis([u, v])
            if dis < minDis:
                minDis = dis
                pair = [u, v]
    return [pair, minDis]


# 分治求解
def divide(seq):
    # 求序列元素数量
    n = len(seq)
    # 按点的纵坐标升序排序
    seq = sorted(seq)
    # 递归开始进行
    if n <= 1:
        return None, float('inf')
    elif n == 2:
        return [seq, calDis(seq)]
    else:
        half = int(len(seq) / 2)
        med_x = (seq[half][0] + seq[-half - 1][0]) / 2
        left = seq[:half]
        resLeft = divide(left)
        right = seq[half:]
        resRight = divide(right)
        # 获取两集合中距离最短的点对
        if resLeft[1] < resRight[1]:
            resMin = combine(left, right, resLeft, med_x)
        else:
            resMin = combine(left, right, resRight, med_x)
        pair = resMin[0]
        minDis = resMin[1]
    return [pair, minDis]



global point1, point2, img

#用于鼠标点击拖拽计算比例尺
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 5)
        cv2.imshow('img', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
        # cv2.rectangle(img2, point1, (x,y), (255,0,0), 5)
        cv2.arrowedLine(img2,point1,(x,y),(0,255,0),5)
        # print(x,y)
        cv2.imshow('img', img2)

    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        # cv2.rectangle(img2, point1, point2, (0,0,255), 5)
        cv2.arrowedLine(img2, point1, (x, y), (0, 255, 0), 5)
        cv2.imshow('img', img2)






def apply_mask(image, mask, color, alpha=0.5):
    """
    打上mask图标
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), 
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):

    #计算比例尺的两点以及原图
    global point1, point2, img
    img=image

    # instance的数量
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = colors or random_colors(N)

    # 当masked_image为原图时是在原图上绘制
    # 如果不想在原图上绘制，可以把masked_image设置成等大小的全0矩阵
    masked_image = np.array(image,np.uint8)

    # ---------------------------------------------------#
    # 获取图片对应的比例尺
    # ---------------------------------------------------#
    # resize image
    width = int(img.shape[1])
    height = int(img.shape[0])

    scale0 = 1.
    while True:  # 图片降维不然显示不了...
        if width <= 800 or height <= 800:
            break
        width = width / 4 * 3
        height = height / 4 * 3
    scale0 = image.shape[1]/width
    dim = (int(width), int(height))
    print('scale0=', scale0)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', on_mouse)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    truelength = input('输入参照物长度（cm）:')
    try:
        truelength = float(truelength)
    except:
        print('Open Error! Try again!')
        truelength = input('输入参照物长度（cm）:')
    else:
        truelength = float(truelength)
    points_dis = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    global scale
    scale = float(truelength)/(points_dis*scale0)
    print('原图大小为：', image.shape)
    print('resize之后大小为：',img.shape)
    print('缩放倍数为'+format(scale0,'.0%'))
    print('参照物像素距离：'+str(points_dis)+' pixels')
    print('参照物实际距离：'+str(truelength)+' cm')
    print("比例尺为：" + format(scale, '.3f') + ' cm/pixel')


    # ---------------------------------------------------#
    #对每一个拓扑进行绘制轮廓
    # ---------------------------------------------------#
    for i in range(N):
        color = colors[i]

        # 该部分用于显示bbox
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (color[0] * 255,color[1] * 255,color[2] * 255), 2)

        # # 该部分用于显示文字与置信度
        # if not captions:
        #     class_id = class_ids[i]
        #     score = scores[i] if scores is not None else None
        #     label = class_names[class_id]
        #     caption = "{} {:.3f} {}".format(label, score, ) if score else label
        # else:
        #     caption = captions[i]
        #
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(masked_image, caption, (x1, y1 + 8), font, 1, (255, 255, 255), 2)

        # ---------------------------------------------------#
        # 该部分用于显示语义分割part
        # ---------------------------------------------------#
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # ---------------------------------------------------#
        # 画出语义分割的范围
        # ---------------------------------------------------#
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        MaxDis=0.
        for verts in contours:
            verts = np.fliplr(verts) - 1
            cv2.polylines(masked_image, [np.array([verts],np.int)], 1, (color[0] * 255,color[1] * 255,color[2] * 255), 2)

            # 找到并且画出长轴
            maxdis = 0
            pair = []
            # print(verts)
            pair, maxdis = calDirect(verts)
            print("长轴长度为：" + str(maxdis)+'pixels')
            # print(pair)
            cv2.polylines(masked_image, [np.array([pair], np.int)], 1,
                          (color[0] * 255, color[1] * 255, color[2] * 255), 2)
            MaxDis=maxdis

        # ---------------------------------------------------#
        # 该部分用于显示文字与置信度
        # ---------------------------------------------------#
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]


            #通过比例尺计算长轴的实际大小计算
            MaxDis=MaxDis*scale
            # caption = "{} {:.3f} {:.3f}cm".format(label, score, MaxDis) if score else label
            caption = "{}".format(label)
        else:
            caption = captions[i]


        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(masked_image, caption, (x1, y1 + 8), font, 1, (255, 255, 255), 2)




        



    img = Image.fromarray(np.uint8(masked_image))
    return img