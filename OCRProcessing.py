import cv2
import numpy as np

# Harrisコーナー検出（cornerHarris）
def get_corner(s_file, r_file):
    img = cv2.imread(s_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    img_dst = np.copy(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.05, img_dst)
    dst = cv2.dilate(dst,None,iterations = 3) # ドットを膨張(Dilation)させて見やすくする処理

    img[dst>0.05*dst.max()]=[0,0,255] # 0.05はドット表示の閾値

    cv2.imwrite(r_file, img)

# Shi-Tomasiコーナー検出
def get_feature(s_file, r_file):
    img = cv2.imread(s_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10)
    # corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, minDistance=200)
    # corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.1, minDistance=10)

    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),5,[0, 0, 255],-1)

    cv2.imwrite(r_file, img)