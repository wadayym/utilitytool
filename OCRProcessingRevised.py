import os
import cv2
import numpy as np

import subImageProcessClass as subIP
import subNumberPlaceRevised as subNP
import subONNX as subOnnx

def find_square(s_file, r_file):
    IPClass = subIP.ImageProcess(s_file)

    # 輪郭の抽出
    contours = IPClass.getContours()

    # 輪郭をなめらかな多角形に変換
    polies = IPClass.getSmoothContours(contours)

    # 多角形から矩形の抽出
    square = IPClass.getRectangle(polies)

    # 各グリッドの頂点を元画像にフィットさせる
    rect = square.astype(np.float32)
    imgTrandformed, grayTransformed = IPClass.transformedImage(rect)
    cross_points, grays, results, candidates_img, templates = IPClass.fitGrid(grayTransformed)

    # 十字の中心点を元の画像に変換
    gridImages = IPClass.getGridImages(cross_points, rect)

    # ONNXモデルを読み込む
    ONNXClass = subOnnx.Onnx('./net/model.onnx')
    # 画像から数字を推論し、タイル状画像を作成 (9行9列)
    tile_image = np.full((9*70, 9*70, 3), 100, dtype=np.uint8)
    NPClass = subNP.NumberPlace()
    for j in range(9):
        for i in range(9):
            # 画像を推論
            im = gridImages[i,j]
            im = np.clip(im, 0, 255).astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.bitwise_not(im)
            # ONNXモデルで推論
            max_idx, max_value = ONNXClass.predict(im)
            NPClass.set(j, i, max_idx)
            # タイル状画像に配置
            tile_image[j*70+3:(j+1)*70-3, i*70+3:(i+1)*70-3] = gridImages[i,j]

    # タイル状画像を描画
    NPClass.check_all()
    number_table, input_table = NPClass.get()
    for j in range(9):
        for i in range(9):
            if input_table[j][i] == 0:
                cv2.putText(tile_image, str(number_table[j][i]), tuple([i*70+17, j*70+55]), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 64, 0), 5)
            else:
                cv2.putText(tile_image, str(number_table[j][i]), tuple([i*70+50, j*70+65]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 0, 0), 3)
    cv2.imwrite(r_file, tile_image)
