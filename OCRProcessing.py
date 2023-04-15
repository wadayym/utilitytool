import os
import cv2
import numpy as np
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

#縦横の周期を自己相関から求める
def find_square_size(image, lines, axis):
    line_image = np.zeros_like(image)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        theta = np.rad2deg(np.arctan2(abs(y2-y1),abs(x2-x1)))
        if axis==1:
            if theta < 10:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255))
        else:
            if theta > 80:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255))
    # 線だけの画像で積算プロファイルを作成
    line_image_blur = cv2.GaussianBlur(line_image,(5,5),0)
    line_image_sum = np.sum(line_image_blur, axis=axis)
    # 自己相関で周期を求める
    line_image_sum = line_image_sum - line_image_sum.mean()
    acr = np.correlate(line_image_sum, line_image_sum, mode='full')
    acr = acr[acr.size//2:]

    h, w = image.shape
    grid_size = min(h,w)
    sq_size = int(grid_size/9+0.5)
    return np.argmax(acr[int(sq_size/2):sq_size])+int(sq_size/2),line_image_sum,acr,line_image

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

# 積算プロファイルで1次元のパターンマッチングを行う。
def find_grid(sum,grid_blur,margin,sq_size,axis):
    sum_grid = np.sum(grid_blur, axis=axis)
    ones = np.ones(len(sum_grid))
    sum_mean = np.convolve(sum, ones, mode="valid") 
    point_candidate = np.argmax(sum_mean)
    # そ積算プロファイルで1次元のパターンマッチングを行う。
    sum_grid = sum_grid - sum_grid.mean()
    area_start = max(point_candidate-int(sq_size/2),0)
    area_end = min(area_start+sq_size,len(sum_grid))
    crr = np.correlate(sum, sum_grid)[area_start:area_end]
    point = np.argmax(crr)+area_start+margin # スタート位置
    return point,crr

def find_9x9square(gray,sq_h,sq_w,ver_point,hor_point,margin):
    square = np.zeros((sq_h+margin*2,sq_w+margin*2), dtype = np.uint8) 
    cv2.rectangle(square,(margin,margin),(margin+sq_w,margin+sq_h),(255,255,255))
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
            sq_tops[y,x] = maxLoc[1] + top + margin
            sq_lefts[y,x] = maxLoc[0] + left + margin
    return sq_tops,sq_lefts

# ます目を描画
def draw_squres(sq_tops,sq_lefts,sq_w,sq_h,img):
    colors = [(0,0,255),(0,255,0),(255,0,0)]
    squares = np.copy(img)
    for y in range(9):
        for x in range(9):
            x1 = sq_lefts[y,x]
            y1 = sq_tops[y,x]
            x2 = x1 + sq_w
            y2 = y1 + sq_h
            cv2.rectangle(squares,(x1,y1),(x2,y2),colors[(x+y)%3])
    return squares

# 認識すべき領域を描画
def draw_recog_area(sq_tops,sq_lefts,sq_w,sq_h,img):
    recogs = np.copy(img)
    for y in range(9):
        for x in range(9):
            x1 = sq_lefts[y,x]
            y1 = sq_tops[y,x]
            x2 = x1 + sq_w
            y2 = y1 + sq_h
            cv2.rectangle(recogs,(x1,y1),(x2,y2),255)            
    return recogs

# ます目に数字があるが判定する
def judge_with_digit(sq_tops,sq_lefts,sq_w,sq_h,img):
    recogs = np.copy(img)
    count = np.zeros((9,9), dtype = int) 
    find = np.zeros((9,9), dtype = bool) 
    for y in range(9):
        for x in range(9):
            x1 = sq_lefts[y,x]
            y1 = sq_tops[y,x]
            x2 = x1 + sq_w
            y2 = y1 + sq_h
            count[y,x]=np.count_nonzero(recogs[y1:y2, x1:x2] > 0)
            if count[y,x] > sq_w*sq_h*0.01:
                find[y,x] = True
    print('count & find:')
    print(count)
    print(find)
    return find

# ます目の中の数字をOCRで読み取る。
def recognize_digit(gray,sq_dig,sq_tops,sq_lefts,sq_w,sq_h):
    number_place = np.zeros((9,9), dtype = int) 
    arr=np.full((9,9),'0',dtype=str)
    dst = Image.new('RGB', (sq_w*9, sq_h*9),(127,127,127))
    for y in range(9):
        for x in range(9):
            if sq_dig[y,x]:
                x1 = sq_lefts[y,x]+int(sq_w*0.1)
                y1 = sq_tops[y,x]+int(sq_h*0.1)
                x2 = x1 + int(sq_w*0.8)
                y2 = y1 + int(sq_h*0.8)             
                ret, th = cv2.threshold(gray[y1:y2, x1:x2], 0, 255, cv2.THRESH_OTSU)
                img = Image.fromarray(th)
                str_digit = pytesseract.image_to_string(img, lang='eng', config='--psm 6 --oem 1 -c tessedit_char_whitelist="123456789"')
                arr[y,x] = str_digit
                try:
                    num = int(str_digit)
                    if num < 1 or num > 9:
                        num = -1
                except ValueError:
                    num = -1                             
                number_place[y,x] = num                
                dst.paste(img, (x*sq_w+int(sq_w*0.1), y*sq_h+int(sq_h*0.1)))
    print('number_place')
    print(arr)
    print(number_place)
    return number_place,dst

# 最小二乗法 y=a*x+b 戻り値：a,b
def find_optimized_line(x,y):
    A = np.vstack([x, np.ones(len(x))]).T
    return np.linalg.lstsq(A, y, rcond=None)[0]

# ます目の位置を調整
def tune_square_position(sq_tops,sq_lefts,sq_h,sq_w):
    sq_tops_tuned = np.copy(sq_tops)
    sq_lefts_tuned = np.copy(sq_lefts)
    tops_cross,lefts_cross = fined_cross_point(sq_tops,sq_lefts)
    print(sq_tops)
    print(sq_lefts)
    print(tops_cross)
    print(lefts_cross)
    print(tops_cross-sq_tops)
    print(lefts_cross-sq_lefts)
    for y in range(9):
        for x in range(9):
            if abs(tops_cross[y,x]-sq_tops[y,x])>sq_h*0.1\
            or abs(lefts_cross[y,x]-sq_lefts[y,x])>sq_w*0.1:
                sq_tops_tuned[y,x] = -1
                sq_lefts_tuned[y,x] = -1
    print(sq_tops_tuned)
    print(sq_lefts_tuned)
    tops_cross,lefts_cross = fined_cross_point(sq_tops_tuned,sq_lefts_tuned)
    print(tops_cross)
    print(lefts_cross)
    for y in range(9):
        for x in range(9):
            if abs(tops_cross[y,x]-sq_tops[y,x])>sq_h*0.1\
            or abs(lefts_cross[y,x]-sq_lefts[y,x])>sq_w*0.1:
                sq_tops_tuned[y,x] = tops_cross[y,x]
                sq_lefts_tuned[y,x] = lefts_cross[y,x]
            else:
                sq_tops_tuned[y,x] = sq_tops[y,x]
                sq_lefts_tuned[y,x] = sq_lefts[y,x]
    print(sq_tops_tuned)
    print(sq_lefts_tuned)
    return sq_tops_tuned,sq_lefts_tuned

# ます目の位置の候補を最小二乗法で求める
def fined_cross_point(sq_tops,sq_lefts):
    tops_tuned = np.copy(sq_tops)
    lefts_tuned = np.copy(sq_lefts)
    a = np.zeros(9, dtype = float) 
    b = np.zeros(9, dtype = float) 
    c = np.zeros(9, dtype = float)  
    d = np.zeros(9, dtype = float) 
    for i in range(9):
        x = np.empty(0, dtype=int)
        y = np.empty(0, dtype=int)
        for j in range(9):
            if sq_lefts[i,j] > 0:
                x = np.append(x, sq_lefts[i,j])
            if sq_tops[i,j] > 0:
                y = np.append(y, sq_tops[i,j])
        a[i],b[i] = find_optimized_line(x,y) # y=a*x+b
    for i in range(9):
        x = np.empty(0, dtype=int)
        y = np.empty(0, dtype=int)
        for j in range(9):
            if sq_lefts[j,i] > 0:
                x = np.append(x, sq_lefts[j,i])
            if sq_tops[j,i] > 0:
                y = np.append(y, sq_tops[j,i])
        c[i],d[i] = find_optimized_line(y,x) # x=c*y+d
    for i in range(9):
        for j in range(9):
            lefts_tuned[j,i] = (b[j]*c[i]+d[i])/(1-a[j]*c[i]) 
            tops_tuned[j,i] = (a[j]*d[i]+b[j])/(1-a[j]*c[i]) 
    return tops_tuned,lefts_tuned

def  overlay_result(img, number_place, sq_tops_tuned,sq_lefts_tuned):
    img_recognized = img.copy()
    text = "0"
    fontFace = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    fontScale = 2
    thickness = 3
    baseline = 0
    (w, h), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    baseline += thickness
    for i in range(9):
        for j in range(9):
             text = str(number_place[j,i])
             if text == "0":
                 text = "."
             cv2.putText(img_recognized, text, [sq_lefts_tuned[j,i], sq_tops_tuned[j,i] + h], fontFace, fontScale, [255,255,255], thickness, 8)
    return img_recognized

# ます目検出
def find_square(s_file, r_file):
    img = cv2.imread(s_file)
    gray = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    
    #縦横の周期を自己相関から求める
    sq_w,ver_sum,v_acr,ver_line_image = find_square_size(gray,lines,axis=0)
    sq_h,hor_sum,h_acr,hor_line_image = find_square_size(gray,lines,axis=1)
    print("size:",sq_h,sq_w)

    # 求めた周期で9x9のます目画像を作成
    margin = 3
    grid_blur = make_grid(sq_h,sq_w,margin)

    # ます目画像とgridの積算プロファイルを作成し、相互相関でマッチングする位置を検出する。
    ver_point,h_ccr = find_grid(hor_sum,grid_blur,margin,sq_h,axis=1)
    hor_point,v_ccr = find_grid(ver_sum,grid_blur,margin,sq_w,axis=0)
    print('point:', ver_point, hor_point)

    # ます目を1個づつ（計9x9回）、それを含む30%広いエリアで1ます目をぼかした画像とパターンマッチングを行う。
    sq_tops,sq_lefts = find_9x9square(gray,sq_h,sq_w,ver_point,hor_point,margin)

    # 9x9個のます目の位置を調整する。最小二乗法でグリッドの線を求め、そこから外れたます目の位置を調整する。
    sq_tops_tuned,sq_lefts_tuned = tune_square_position(sq_tops,sq_lefts,sq_h,sq_w)
    squares_tuned =  draw_squres(sq_tops_tuned,sq_lefts_tuned,sq_w,sq_h,img)

    # 数字があるます目を見つける。
    recog_area1 = draw_recog_area(sq_tops_tuned+int(sq_h*0.1),sq_lefts_tuned+int(sq_w*0.1),int(sq_w*0.8),int(sq_h*0.8),edges)
    recog_area2 = draw_recog_area(sq_tops_tuned+int(sq_h*0.2),sq_lefts_tuned+int(sq_w*0.2),int(sq_w*0.6),int(sq_h*0.6),recog_area1)
    sq_dig = judge_with_digit(sq_tops_tuned+int(sq_h*0.2),sq_lefts_tuned+int(sq_w*0.2),int(sq_w*0.6),int(sq_h*0.6),edges)

    # ます目の中の数字をOCRで読み取る。
    number_place,image_for_recog = recognize_digit(gray,sq_dig,sq_tops_tuned,sq_lefts_tuned,sq_w,sq_h)

    #　読み取った数字をオーバレイする。
    image_recognised = overlay_result(img, number_place, sq_tops_tuned,sq_lefts_tuned)

    file, ext = os.path.splitext(r_file)
    save_graph(file+'_sum'+ext,ver_sum,hor_sum)
    save_graph(file+'_acr'+ext,v_acr,h_acr)
    save_graph(file+'_ccr'+ext,v_ccr,h_ccr)
    cv2.imwrite(file+'_cross'+ext, edges)
    cv2.imwrite(file+'_grid_blur'+ext, grid_blur)
    cv2.imwrite(file+'_recog_area'+ext,recog_area2)
    cv2.imwrite(file+'_line'+ext,cv2.hconcat([ver_line_image, hor_line_image]))
    cv2.imwrite(file+'_squares'+ext, squares_tuned)
    cv2.imwrite(r_file,np.array(image_recognised))
    # cv2.imwrite(r_file,np.array(image_for_recog))


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
