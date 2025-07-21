import sys
import numpy as np
import itertools
import cv2

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
        #self.gray = np.max(self.img, axis=2).astype(np.uint8)  # RGBの最大値をとる

        edgeImage = cv2.Canny(self.gray, 96, 128)
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        for filledImg in filledImages:
            #setImg = cv2.morphologyEx(filledImg, cv2.MORPH_OPEN, kernel)
            setimg0 = cv2.erode(filledImg, kernel, iterations=4)
            setImg = cv2.dilate(setimg0, kernel, iterations=3)
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

    def transformedImage(self, rect):
        # 元画像を正規化する
        grid_size, margin, grid_pnts = self.getGridPoints() # グリッドマトリックス
        #print("rect type:", rect.dtype)
        #print("grid_pnts type:", grid_pnts.dtype)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)  
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
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
        kernel = cv2.resize(kernel, (31, 31), interpolation=cv2.INTER_AREA)
        kernel = cv2.GaussianBlur(kernel, ksize=(3, 3), sigmaX=0, borderType = cv2.BORDER_REPLICATE) 
        bottom = kernel / np.sum(kernel)  # 正規化
        templates = [center, left_top, right_top, left_bottom, right_bottom, left, right, top, bottom]
        
        cross_points = np.zeros((10, 10, 2), dtype=np.int32)
        cross_points_candidate = [[[] for _ in range(10)] for _ in range(10)]
        grays = np.empty((10, 10, 64, 64), dtype=np.uint8)
        results = np.empty((10, 10, 34, 34), dtype=np.float32)
        candidates_img = np.empty((10, 10, 34, 34, 3), dtype=np.uint8)    

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
                center_x = result.shape[1] // 2
                center_y = result.shape[0] // 2                
                neighbor = np.zeros(result.shape, dtype=np.int32)
                threshold = np.max(result) * 0.8  # スコアの閾値
                # 閾値以上の候補を抽出
                locations = np.where(result >= threshold)
                # 座標とスコアをタプルで保持
                matches = [(pt[1], pt[0], result[pt[0], pt[1]]) for pt in zip(*locations)]
                # スコアでソート（高い順）
                sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
                #print(sorted_matches[:10])  # 上位10件を表示
                # スコアの高い順に処理
                # 極大値だけを候補とする　5x5の領域をマークして極大値でないものを除外
                for x, y, score in sorted_matches:
                    if neighbor[y, x] == 0:
                        cross_points_candidate[i][j].append((x, y, score))
                    ys = y - 2
                    if ys < 0:
                        ys = 0
                    xs = x - 2
                    if xs < 0:
                        xs = 0
                    ye = y + 2
                    if ye >= result.shape[0]:
                        ye = result.shape[0] - 1
                    xe = x + 2
                    if xe >= result.shape[1]:
                        xe = result.shape[1] - 1
                    neighbor[ys:ye + 1, xs:xe + 1] = 1  # 5x5の領域をマーク
                #print(f"グリッド({i}, {j})の候補数: {len(cross_points_candidate[i][j])}")
                #print(cross_points_candidate[i][j])
                candidates_img[i,j] = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                for c in cross_points_candidate[i][j]:
                    cv2.circle(candidates_img[i,j], (c[0], c[1]), 3, (0, 255, 0), -1)
                results[i, j] = result
                grays[i, j] = gray_blurred[j * 64: j * 64 + 64, i * 64: i * 64 + 64]

        # AdjustmentOrderに従ってcross_pointsを調整する
        AdjustmentOrder = np.array([4, 5, 3, 6, 2, 7, 1, 8, 0, 9])  # 調整順序        
        AdjustmentTimes = 5  # 調整回数
        for k in range(AdjustmentTimes): 
            for j in AdjustmentOrder:
                for i in AdjustmentOrder:
                    # 上下左右の点を取得
                    left  = cross_points[i - 1, j] if i > 0 else np.array([0, 0])
                    right = cross_points[i + 1, j] if i < 9 else np.array([0, 0])
                    up    = cross_points[i, j - 1] if j > 0 else np.array([0, 0])
                    down  = cross_points[i, j + 1] if j < 9 else np.array([0, 0])
                    # 平均をとる
                    mean_point = np.mean([up, down, left, right], axis=0)
                    line_lens = [np.linalg.norm(np.array([c[0] - center_x - mean_point[0], c[1] - center_y - mean_point[1]])) for c in cross_points_candidate[i][j]]
                    min_index = line_lens.index(min(line_lens))
                    cross_points[i, j] = tuple(x + y for x, y in zip((-center_x, -center_y), cross_points_candidate[i][j][min_index][:2])) 
                    if k == AdjustmentTimes - 1:  # 最後の調整であれば、中心点を青くする
                        cv2.circle(candidates_img[i,j],                       
                                (cross_points[i, j][0] + center_x, cross_points[i, j][1] + center_y),
                                5, (255, 0, 0), 2)
                    else:  # 調整中は赤くする
                        cv2.circle(candidates_img[i,j],                      
                                (cross_points[i, j][0] + center_x, cross_points[i, j][1] + center_y),
                                5, (0, 0, 255), 2)

        # 十字の中心点をマージンを加えて配置
        for j in range(10):
            for i in range(10):
                cross_points[i, j] += np.array([i * 64 + 32, j * 64 + 32])  # マージンを加える
         
        return cross_points, grays, results, candidates_img, templates

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