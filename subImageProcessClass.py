import os
import sys
import numpy as np
import itertools
import cv2
from PIL import Image
from scipy.optimize import basinhopping

class ImageProcess:
    def __init__(self, file_name):
        # コンストラクタ：オブジェクトの初期化
        self.raw = cv2.imread(file_name)

        # 320x320に外接するサイズにリサイズする
        size = self.raw.shape[:2]
        h, w = 320, 320
        f = max(h / size[0], w / size[1])
        self.img = cv2.resize(self.raw, (int(size[1] * f), int(size[0] * f)), interpolation=cv2.INTER_AREA)

        # convert to gray
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        edgeImage = cv2.Canny(self.gray, 50, 128)
        # Closing morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        self.edge = cv2.morphologyEx(edgeImage, cv2.MORPH_CLOSE, kernel)
        
    def getRaw(self):
        return self.raw

    def getImg(self):
        return self.img

    def getGray(self):
        return self.gray
    
    def getEdge(self):
        return self.edge

    def getRaw(self):
        return self.raw

    def getContours(self):
        contours, hierarchy = cv2.findContours(self.edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.edge.shape[0] * self.edge.shape[1] * 0.2 # 画像の何割占めるか
        large_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                large_contours.append(c)
        return large_contours

    def getSmoothContours(self, contours):
        filledImages = []
        for contour in contours:
            blank = np.zeros(self.gray.shape, np.uint8)
            cv2.drawContours(blank, [contour], -1, 255, -1)
            filledImages.append(blank)

        # 輪郭のノイズを除去する
        setImages = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        for filledImg in filledImages:
            setImg = cv2.morphologyEx(filledImg, cv2.MORPH_OPEN, kernel)
            setImages.append(setImg)

        # contours 再度検出
        smooth_contours = []
        for setImg in setImages:
            contours, hierarchy = cv2.findContours(setImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            min_area = setImg.shape[0] * setImg.shape[1] * 0.2 # 画像の何割占めるか
            for c in contours:
                area = cv2.contourArea(c)
                if area > min_area:
                    smooth_contours.append(c)

        # contours　を凸にする
        convexes = []
        for contour in smooth_contours:
            convex = cv2.convexHull(contour)
            convexes.append(convex)
        polies = []
        for contour in convexes:
            arclen = cv2.arcLength(contour, True)
            poly = cv2.approxPolyDP(contour, 0.005*arclen, True)
            # 近似した輪郭を2次元の座標に変換してpoliesに追加
            polies.append(poly[:, 0, :])
        return polies

    def getRectangle(self, polies):
        # poliesから最も正方形に近い矩形を検出する
        score_min = sys.float_info.max
        square = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)
        for poly in polies:
            choices = np.array(list(itertools.combinations(poly, 4)))
            # 正方形に近いものを選ぶ
            for c in choices:
                # 右回りに並べ替える
                c = np.array(c)
                c = c[np.argsort(np.arctan2(c[:, 1] - np.mean(c[:, 1]), c[:, 0] - np.mean(c[:, 0])))]
                # 各辺の長さを計算
                line_lens = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
                base = cv2.contourArea(c) ** 0.5
                score = sum([abs(1 - l/base) ** 2 for l in line_lens])
                if score < score_min:
                    score_min = score
                    square = c
        return square

    def getLineImage(self, square):
        # 線分化
        poly_length = cv2.arcLength(square, True)
        threshold = int(poly_length / 12)
        minLineLength = int(poly_length / 200)
        maxLineGap = 5
        if threshold < 1:
            threshold = 1
        if minLineLength < 1:
            minLineLength = 1
        lines = cv2.HoughLinesP(self.edge, 1, np.radians(1), threshold, minLineLength, maxLineGap)
        line_mat = np.zeros(self.edge.shape, np.uint8)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_mat, (x1, y1), (x2, y2), 255, 1)
            
        # 矩形の外をマスクアウト
        mask = np.zeros(line_mat.shape, np.uint8)
        cv2.fillConvexPoly(mask, square, 1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=3)
        line_mat[np.where(mask == 0)] = 0
        return line_mat

    # スコアマトリックスを生成する
    def gen_score_mat(self):
        # スコアマトリックスを生成
        cell = np.fromfunction(
            lambda x, y: np.maximum((10 - x) ** 2, (10 - y) ** 2) / 100.0,
            (21, 21),
            dtype=np.float32
        )
        score_mat = np.zeros((209, 209), np.float32)
        for i in range(209):
            for j in range(209):
                l = min(i, j, 208 - i, 208 - j)
                if l > 10:
                    continue
                score_mat[i, j] = l*l/100.0
        score_mat[10:199, 10:199] = np.tile(cell, (9, 9))
        return score_mat

    def get_get_fit_score(self,line_mat, score_mat):
        def get_fit_score(x):
            img_pnts = np.float32(x).reshape(4, 2)
            score_size = score_mat.shape[0]
            score_pnts = np.float32([[10, 10], [10, score_size-10], [score_size-10, score_size-10], [score_size-10, 10]])
            transform = cv2.getPerspectiveTransform(score_pnts, img_pnts)
            score_t = cv2.warpPerspective(score_mat, transform, (self.edge.shape[1], self.edge.shape[0]))
            # スコアを計算
            res = line_mat * score_t
            return -np.average(res[np.where(res > 255 * 0.1)])
            #return -np.sum(res)
        return get_fit_score
    
    def fitRectangle(self, square, line_mat):
        score_mat = self.gen_score_mat()
        get_fit_score = self.get_get_fit_score(line_mat, score_mat)
        x0 = square.flatten()
        ret = basinhopping(get_fit_score, x0, T=0.1, niter=300, stepsize=3)
        rect = ret.x.reshape((4, 2)).astype(np.float32)
        score = ret.fun
        return rect, score

    def transformedImage(self, rect):
        # 元画像を正規化する
        grid_size, margin, grid_pnts = self.getGridPoints() # グリッドマトリックス
        print("rect type:", rect.dtype)
        print("grid_pnts type:", grid_pnts.dtype)
        transform = cv2.getPerspectiveTransform(rect, grid_pnts)
        gray_transformed = cv2.warpPerspective(self.gray, transform, (grid_size + margin * 2, grid_size + margin * 2))
        gray_reverted = cv2.bitwise_not(gray_transformed)
        gray_blurred = cv2.GaussianBlur(gray_reverted, (5, 5), 0)
        img_transformed = cv2.warpPerspective(self.img, transform, (grid_size + margin * 2, grid_size + margin * 2))
        return img_transformed, gray_blurred

    def getGridPoints(self):
        grid_size = 64*9  # グリッドマトリックスのサイズ
        margin = 32  # グリッドマトリックスのマージン
        grid_pnts = np.array([[0, 0], [grid_size, 0], [grid_size, grid_size], [0, grid_size]], dtype=np.float32)
        grid_pnts += margin  # マージンを追加
        return grid_size, margin, grid_pnts

    def fitGrid(self, gray_blurred):
        # templateを用意する
        # 中央
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        center = kernel / np.sum(kernel)  # 正規化
        # 左上
        kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        left_top = kernel / np.sum(kernel)  # 正規化
        # 右上
        kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        right_top = kernel / np.sum(kernel)  # 正規化
        # 左下
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        left_bottom = kernel / np.sum(kernel)  # 正規化
        # 右下
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        right_bottom = kernel / np.sum(kernel)  # 正規化
        # 左
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)  
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        left = kernel / np.sum(kernel)  # 正規化
        #left = np.ones(grayImage.shape, np.uint8)*255
        # 右
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        right = kernel / np.sum(kernel)  # 正規化
        #right = np.ones(grayImage.shape, np.uint8)*255
        # 上
        kernel = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        top = kernel / np.sum(kernel)  # 正規化
        # 下
        kernel = np.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        kernel = cv2.resize(kernel, (21, 21), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        bottom = kernel / np.sum(kernel)  # 正規化

        cross_points = np.empty((10, 10, 2), dtype=np.int32)
        # テンプレートマッチングで十字の中心点を検出
        for j in range(10):
            for i in range(10):
                x = i * 64
                y = j * 64
                template = center
                if i == 0 and j == 0:
                    template = left_top
                elif i == 9 and j == 0: # 右上
                    template = right_top        
                elif i == 0 and j == 9: # 左下
                    template = left_bottom
                elif i == 9 and j == 9: # 右下
                    template = right_bottom
                elif i == 0: # 左
                    template = left
                elif i == 9: # 右
                    template = right
                elif j == 0: # 上
                    template = top
                elif j == 9: # 下
                    template = bottom
                # テンプレートマッチング
                result = cv2.matchTemplate(gray_blurred[y:y+64, x:x+64].astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
                # 最大値の位置を取得
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)       
                # 十字の中心点を更新
                offset_x = template.shape[1] // 2
                offset_y = template.shape[0] // 2
                cross_points[i, j] = (max_loc[0] + x + offset_x, max_loc[1] + y + offset_y)
        return cross_points

    def getGridImages(self, cross_points, rect):    
        # 十字の中心点を元の画像に変換
        grid_size, margin, grid_pnts = self.getGridPoints()
        transform = cv2.getPerspectiveTransform(grid_pnts, rect)
        pts = cross_points.reshape(-1, 1, 2).astype(np.float32)
        cross_points_original = cv2.perspectiveTransform(pts, transform)
        cross_points_raw = cross_points_original * (self.raw.shape[0] / self.img.shape[0])  # 元の画像のサイズに変換
        gridImages = np.empty([10, 10, 64, 64, 3], dtype=np.uint8) 
        # グリッドの画像を抽出する
        grid = np.float32([[0, 0], [0, 64], [64, 64], [64, 0]])
        cross_points_raw = cross_points_raw.reshape(10, 10, 2)
        for i in range(9):
            for j in range(9):
                rect = np.float32([
                    cross_points_raw[i, j],
                    cross_points_raw[i, j + 1],
                    cross_points_raw[i + 1, j + 1],
                    cross_points_raw[i + 1, j],
                ])
                transform = cv2.getPerspectiveTransform(rect, grid)
                gridImage = cv2.warpPerspective(self.raw, transform, (64, 64))
                # タイル状画像に配置
                gridImages[i, j] = gridImage
        return gridImages