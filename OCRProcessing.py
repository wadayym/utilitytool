import os
import cv2
import numpy as np
import math
from statistics import mean
from PIL import Image, ImageSequence
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#縦横の周期を求める
def find_square_height_width(image, lines, is_height):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        theta = np.rad2deg(np.arctan2(abs(y2-y1),abs(x2-x1)))
        if is_height:
            if theta < 10:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255))
        else:
            if theta > 80:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255))
    # 線だけの画像で積算プロファイルを作成
    line_image_blur = cv2.GaussianBlur(line_image,(5,5),0)
    if is_height:
        line_image_sum = np.sum(line_image_blur, axis=1)
    else:
        line_image_sum = np.sum(line_image_blur, axis=0)
    # 自己相関で周期を求める
    line_image_sum = line_image_sum - line_image_sum.mean()
    acr = np.correlate(line_image_sum, line_image_sum, mode='full')
    acr = acr[acr.size//2:]

    h, w = image.shape
    grid_size = min(h,w)
    sq_size = int(grid_size/9+0.5)
    return np.argmax(acr[int(sq_size/2):sq_size])+int(sq_size/2),line_image_sum

#グリッド画像を作成
def make_grid(h,w,margin):
    grid_size_w = w*9
    grid_size_h = h*9
    margin = 3
    grid = np.zeros((grid_size_h+margin*2,grid_size_w+margin*2), dtype = np.uint8)    
    for i in range(10):
        x1 = int(margin + i*w)
        y1 = int(margin)
        x2 = x1
        y2 = int(y1 + 9*h)
        cv2.line(grid,(x1,y1),(x2,y2),(255,255,255))
        x1 = int(margin)
        y1 = int(margin + i*h)
        x2 = int(x1 + 9*w)
        y2 = y1
        cv2.line(grid,(x1,y1),(x2,y2),(255,255,255))
    # 少しぼかす。
    grid_image = cv2.GaussianBlur(grid,(5,5),0)
    return grid_image

# ます目検出
def find_square(s_file, r_file):
    img = cv2.imread(s_file)
    gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    sq_w,ver_sum = find_square_height_width(gray,lines,False)
    sq_h,hor_sum = find_square_height_width(gray,lines,True)
    print("size",sq_h,sq_w)

    # 求めた周期で9x9のます目画像を作成
    margin = 3
    grid_blur = make_grid(sq_h,sq_w,margin)
    # ます目画像の積算プロファイルを作成
    ver_sum_grid = np.sum(grid_blur, axis=0)
    hor_sum_grid = np.sum(grid_blur, axis=1)
    # それぞれの積算プロファイルで1次元のパターンマッチングを行う。
    ver_sum_grid = ver_sum_grid - ver_sum_grid.mean()
    hor_sum_grid = hor_sum_grid - hor_sum_grid.mean()
    v_crr = np.correlate(ver_sum, ver_sum_grid)
    h_crr = np.correlate(hor_sum, hor_sum_grid)
    hor_point = np.argmax(v_crr)+margin # 縦方向の積算プロファイルのずれ量が水平方向のスタート位置
    ver_point = np.argmax(h_crr)+margin
    print('point:', ver_point, hor_point)
    # ます目のおおよその位置を確定する。
    # ます目を1個づつ（計9x9回）、それを含む30%広いエリアで1ます目をぼかした画像とパターンマッチングを行う。
    square = np.zeros((sq_h+margin*2,sq_w+margin*2), dtype = np.uint8) 
    x1 = margin
    y1 = margin
    x2 = x1+sq_w
    y2 = y1+sq_h
    cv2.rectangle(square,(x1,y1),(x2,y2),(255,255,255))
    square_blur = cv2.GaussianBlur(square, (5, 5),0)
    sq_tops = np.zeros((9,9), dtype = int) 
    sq_lefts = np.zeros((9,9), dtype = int)  
    h,w = gray.shape
    for y in range(9):
        top = max(min(int(ver_point+sq_h*y-sq_h*0.3+0.5),int(h-sq_h*1.6)),0)
        for x in range(9):
            left = max(min(int(hor_point+sq_w*x-sq_w*0.3+0.5),int(w-sq_w*1.6)),0)
            target_img = gray[top:top+int(sq_h*1.6),left:left+int(sq_w*1.6)]
            result = cv2.matchTemplate(target_img, square_blur, cv2.TM_CCOEFF_NORMED)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
            print('top:',top,'  left:',left)
            sq_tops[y,x]=maxLoc[1]+top
            sq_lefts[y,x]=maxLoc[0]+left
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    squares = np.copy(img)
    for y in range(9):
        for x in range(9):
            x1 = sq_lefts[y,x] + margin
            y1 = sq_tops[y,x] + margin
            x2 = x1 + sq_w
            y2 = y1 + sq_h
            cv2.rectangle(squares,(x1,y1),(x2,y2),colors[(x+y)%3])
    # 9x9個のます目の位置を調整する。最小二乗法でグリッドの線を求め、そこから外れたます目の位置を調整する。

    # ます目の中の数字をOCRで読み取る。

    file, ext = os.path.splitext(r_file)
    save_graph(file+'_crr'+ext, v_crr, h_crr)
    save_graph(file+'_vsum'+ext, ver_sum, ver_sum_grid)
    save_graph(file+'_hsum'+ext, hor_sum, hor_sum_grid)
    cv2.imwrite(file+'_cross'+ext, edges)
    cv2.imwrite(file+'_grid_blur'+ext, grid_blur)
    cv2.imwrite(r_file, squares)

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


 
def save_graph(filename_save, x_list, y_list):
    fig = plt.figure(figsize=(8, 6), dpi=300)
 
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0
    i_list = list(range(len(x_list)))

    fig1 = fig.add_subplot(1, 2, 1)
    fig1.plot(i_list, x_list, color = '#008000', linestyle = "-")
    #fig1.set_ylim(x_ave - 15, x_ave + 15)
    plt.title("x_list")
    #fig1.tick_params(labelsize=20)

    i_list = list(range(len(y_list)))
    fig2 = fig.add_subplot(1, 2, 2)
    fig2.plot(i_list, y_list, color = '#000080', linestyle = "-")
    #fig2.set_ylim(y_ave - 15, y_ave + 15)
    plt.title("y_list")
    #fig2.tick_params(labelsize=20)
 
    plt.rcParams["font.size"] = 11
    plt.tight_layout()
 
    plt.show()
    plt.savefig(filename_save)
    return
