import os
import cv2
import numpy as np
import math
from statistics import mean

# ます目検出
def find_square(s_file, r_file):
    img = cv2.imread(s_file)
    h, w, c = img.shape
    grid_size = min(h,w)
    sq_size = int(grid_size/9+0.5)
    gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    ver = np.zeros_like(gray)
    hor = np.zeros_like(gray)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        theta = np.rad2deg(np.arctan2(abs(y2-y1),abs(x2-x1)))
        if theta > 80:
            cv2.line(ver,(x1,y1),(x2,y2),(255,255,255))
        if theta < 10:
            cv2.line(hor,(x1,y1),(x2,y2),(255,255,255))
    # 縦線だけの画像　で垂直方向の積算プロファイルを作成
    ver_sum = np.sum(ver, axis=0)
    # 横線だけの画像　で水平方向の積算プロファイルを作成
    hor_sum = np.sum(hor, axis=1)
    # それぞれの自己相関で周期を求める
    ver_sum = ver_sum - ver_sum.mean()
    result = np.correlate(ver_sum, ver_sum, mode='full')
    result = result[result.size//2:]
    ver_cycle = np.argmax(result[int(sq_size/2):sq_size])+int(sq_size/2)
    r_img = np.zeros((sq_size,sq_size), dtype = np.uint8)
    result = np.int16(result/np.max(result)*sq_size)
    for i in range(sq_size):
        cv2.circle(r_img, (i,result[i]), 2, (255,255,255), thickness=-1)
  
    hor_sum = hor_sum - hor_sum.mean()
    result = np.correlate(hor_sum, hor_sum, mode='full')
    result = result[result.size//2:]
    hor_cycle = np.argmax(result[int(sq_size/2):sq_size])+int(sq_size/2)
    cycle = mean([ver_cycle,hor_cycle])
    print("cycle",ver_cycle,hor_cycle,cycle,sq_size)
    # その周期の平均で9x9のます目画像を作成
    sq_size = cycle
    grid_size = sq_size*9
    margin = 3
    grid = np.zeros((grid_size+margin*2,grid_size+margin*2), dtype = np.uint8)    
    for i in range(10):
        x1 = int(margin + i*sq_size)
        y1 = int(margin)
        x2 = x1
        y2 = int(y1 + 9*sq_size)
        cv2.line(grid,(x1,y1),(x2,y2),(255,255,255))
        cv2.line(grid,(y1,x1),(y2,x2),(255,255,255))
    # 少しぼかす。
    grid_blur = cv2.blur(grid, (3, 3))
    imgs = cv2.hconcat([grid, grid_blur])
    # ます目画像の積算プロファイルを作成
    ver_sum_grid = np.sum(grid_blur, axis=0)
    hor_sum_grid = np.sum(grid_blur, axis=1)
    # それぞれの積算プロファイルで1次元のパターンマッチングを行う。
    ver_sum = ver_sum - ver_sum.mean()
    ver_sum_grid = ver_sum_grid - ver_sum_grid.mean()
    hor_sum = hor_sum - hor_sum.mean()
    hor_sum_grid = hor_sum_grid - hor_sum_grid.mean()
    ver_point = np.argmax(np.correlate(ver_sum, ver_sum_grid))+margin
    hor_point = np.argmax(np.correlate(hor_sum, hor_sum_grid))+margin
    print('point:', ver_point, hor_point)
    # ます目のおおよその位置を確定する。
    # ます目を1個づつ（計9x9回）、それを含む30%広いエリアで1ます目をぼかした画像とパターンマッチングを行う。
    square = np.zeros((sq_size+margin*2,sq_size+margin*2), dtype = np.uint8) 
    x1 = margin
    y1 = margin
    x2 = x1+sq_size
    y2 = y1+sq_size
    cv2.rectangle(square,(x1,y1),(x2,y2),(255,255,255))
    square_blur = cv2.blur(square, (3, 3))
    sq_tops = np.zeros((9,9), dtype = int) 
    sq_lefts = np.zeros((9,9), dtype = int)  
    for y in range(9):
        top = max(min(int(ver_point-sq_size*0.3+0.5),h-sq_size),0)
        for x in range(9):
            left = max(min(int(hor_point-sq_size*0.3+0.5),w-sq_size),0)
            target_img = gray[top:top+sq_size,left:left+sq_size]
            result = cv2.matchTemplate(target_img, square_blur, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            sq_tops[y,x]=maxLoc[1]+top
            sq_lefts[y,x]=maxLoc[0]+left
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    squares = np.copy(img)
    for y in range(9):
        for x in range(9):
            cv2.rectangle(squares,(x1,y1),(x2,y2),colors[(x+y)%3])
    # 9x9個のます目を作成し、その中の数字をOCRで読み取る。

    file, ext = os.path.splitext(r_file)
    cv2.imwrite(file+'_cross'+ext, edges)
    cv2.imwrite(file+'_ver'+ext, ver)
    cv2.imwrite(file+'_hor'+ext, hor)
    cv2.imwrite(file+'_grid_blur'+ext, grid_blur)
    cv2.imwrite(file+'_r_img'+ext, r_img)
    cv2.imwrite(file+'_squares'+ext, squares)
    cv2.imwrite(r_file, gray)

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
    kernel1 = np.zeros((size,size), dtype=np.float32)
    xs = np.arange(-radius,radius+1)
    ys = np.arange(-radius,radius+1)
    for x in xs:
        for y in ys:
            kernel1[y+radius][x+radius] = math.exp(-x*x/sigma2) + math.exp(-y*y/sigma2) - 1.0
    kernel2 = np.array(
        [[0,1,2,3,4,5,4,3,2,1,0],
         [1,0,1,2,3,5,3,2,1,0,1],
         [2,1,0,1,2,5,2,1,0,1,2],
         [3,2,1,0,1,5,1,0,1,2,3],
         [4,3,2,1,0,5,0,1,2,3,4],
         [5,5,5,5,5,5,5,5,5,5,5],
         [4,3,2,1,0,5,0,1,2,3,4],
         [3,2,1,0,1,5,1,0,1,2,3],
         [2,1,0,1,2,5,2,1,0,1,2],
         [1,0,1,2,3,5,3,2,1,0,1],
         [0,1,2,3,4,5,4,3,2,1,0]
         ])
    kernel3 = np.array(
        [[0,0,1,2,3,5,3,2,1,0,0],
         [0,0,0,1,2,5,2,1,0,0,0],
         [1,0,0,1,2,5,1,0,0,0,1],
         [2,1,0,0,1,5,1,0,0,1,2],
         [3,2,1,1,0,5,0,1,1,2,3],
         [5,5,5,5,5,5,5,5,5,5,5],
         [3,2,1,1,0,5,0,1,1,2,3],
         [2,1,0,0,1,5,1,0,0,1,2],
         [1,0,0,1,2,5,1,0,0,0,1],
         [0,0,0,1,2,5,2,1,0,0,0],
         [0,0,1,2,3,5,3,2,1,0,0]
         ])
    kernel = kernel3
    mean = kernel.mean()
    print(mean)
    kernel = kernel - mean
    print(kernel)
    dst = cv2.filter2D(gray, -1, kernel)
    #dst = gray
    min = dst.min()
    max = dst.max()
    cross = np.uint8((dst-min)/(max-min)*255)
    edges = cv2.Canny(cross,50,150,apertureSize = 3)
    file, ext = os.path.splitext(r_file)
    cv2.imwrite(file+'_cross'+ext, edges)

    dst = dst - min
    print(min,max)
    #img[dst>0.6*dst.max()]=[0,0,64] # 0.6はドット表示の閾値 BGR
    #img[dst>0.7*dst.max()]=[0,0,128] # 0.7はドット表示の閾値
    #img[dst>0.8*dst.max()]=[0,0,192] # 0.8はドット表示の閾値
    #img[dst>0.6*dst.max()]=[0,0,255] # 0.9はドット表示の閾値

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
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