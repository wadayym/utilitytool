import cv2
import numpy as np
import math

# 十字検出
def find_cross(s_file, r_file):
    img = cv2.imread(s_file)
    gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    gray = np.float32(gray)

    # フィルタ
    radius = 5
    sigma = 5
    sigma2 = sigma*sigma
    size = 2*radius + 1
    kernel = np.zeros((size,size), dtype=np.float32)
    xs = np.arange(-radius,radius+1)
    ys = np.arange(-radius,radius+1)
    for x in xs:
        for y in ys:
            kernel[y+radius][x+radius] = math.exp(-x*x/sigma2) + math.exp(-y*y/sigma2) - 1.0
    mean = kernel.mean()
    print(mean)
    kernel = kernel - mean
    print(kernel)
    dst = cv2.filter2D(gray, -1, kernel)
    #dst = gray
    min = dst.min()
    dst = dst - min
    print(dst.max())
    img[dst>0.6*dst.max()]=[0,0,64] # 0.05はドット表示の閾値 BGR
    img[dst>0.7*dst.max()]=[0,0,128] # 0.05はドット表示の閾値
    img[dst>0.8*dst.max()]=[0,0,192] # 0.05はドット表示の閾値
    img[dst>0.9*dst.max()]=[0,0,255] # 0.05はドット表示の閾値

    cv2.imwrite(r_file, img)

# Harrisコーナー検出（cornerHarris）
def get_corner(s_file, r_file):
    img = cv2.imread(s_file)
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    img_dst = np.copy(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.05, img_dst)
    dst = cv2.dilate(dst,None,iterations = 3) # ドットを膨張(Dilation)させて見やすくする処理

    img[dst>0.05*dst.max()]=[0,0,255] # 0.05はドット表示の閾値

    cv2.imwrite(r_file, img)

# Shi-Tomasiコーナー検出
def get_feature(s_file, r_file):
    img = cv2.imread(s_file)
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    # corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=200)
    # corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=10)

    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),5,[0, 0, 255],-1)

    cv2.imwrite(r_file, img)