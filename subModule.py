import os
import datetime
DIFF_JST_FROM_UTC = 9 # 日本はUTC+9時間
import psutil
import numpy as np
import cv2
import re
import math
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 現在時刻
def getCurrentTime():
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)
    return now.strftime('%Y-%m-%d %H:%M:%S')

# 画像にラベルをインポーズする
def inposeLabelOnImage(im, pred, ans, value):
    #label = (f"p:{pred}",f"a:{ans}",f"s:{value:.2f}")
    label = (f"{pred}",)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # 緑色
    thickness = 2
    im_color = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # グレースケールからBGRに変換
    # 背景画像と同じサイズの空の画像を作成
    text_image = np.zeros_like(im_color)
    for i, text in enumerate(label):
        # テキストの位置を計算
        text_x = 3
        text_y = 20 * (i + 1)
        # テキストを画像に描画
        cv2.putText(text_image, text, (text_x, text_y), font, font_scale, color, thickness)
    # テキストを元の画像に合成
    im_imposed = cv2.addWeighted(im_color, 0.4, text_image, 0.6, 0)
    return im_imposed

# openCVフォーマットの画像をpillowで表示する
def showImage(cv2_image):
    # BGRからRGBに変換
    img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # PillowのImageオブジェクトに変換
    img_pillow = Image.fromarray(img_rgb)
    # Pillowを使って画像を表示
    img_pillow.show()

# 1チャンネルのNumPy配列をPillowで表示する
def showGrayscaleImage(np_image):
    # NumPy配列をPillowのImageオブジェクトに変換（モード'L'を指定）
    img_pillow = Image.fromarray(np_image, mode='L')
    # Pillowを使って画像を表示
    img_pillow.show()

# 画像をリサイズする。余白は黒で埋める。
def resizeImage (image, size):
    h, w = image.shape[:2]
    aspect = w / h
    nh = nw = size
    if 1 >= aspect:
        nw = round(nh * aspect)
    else:
        nh = round(nw / aspect)

    resized = cv2.resize(image, dsize=(nw, nh), interpolation=cv2.INTER_LANCZOS4)

    h, w = resized.shape[:2]
    x = y = 0
    if h < w:
        y = (size - h) // 2
    else:
        x = (size - w) // 2

    resized = Image.fromarray(resized) #一旦PIL形式に変換
    canvas = Image.new(resized.mode, (size, size), 0)
    canvas.paste(resized, (x, y))

    dst = np.array(canvas) #numpy(OpenCV)形式に戻す
    
    return dst

# 対象を縮小し、回転する。画像のサイズはそのままで、余白は黒で埋める。
def rotateAndresizeObject (image, angle, ratio):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, ratio)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

    return rotated
# 画像を平行移動する。画像のサイズはそのままで、余白は黒で埋める。
def augmentImageByMove (original_image_path, base_dir, Image_move_pixels):
    # make directories for estimation data
    os.makedirs(base_dir, exist_ok=True)
    for x_move_pixels in Image_move_pixels:
        for y_move_pixels in Image_move_pixels:
            os.makedirs(os.path.join(base_dir,f'Xmove_{x_move_pixels:+03d}_Ymove_{y_move_pixels:+03d}'), exist_ok=True)
    # original画像をloadする
    folder_path = original_image_path
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    file_list.sort()
    print(file_list)
    digit_images = []
    for fname in file_list:
        image = Image.open(os.path.join(folder_path, fname))
        image = np.array(image)
        #print(image.shape)
        #print('data type',type(image))
        digit_images.append(image)
    # 画像の数を取得
    num_images = len(file_list)
    num_rows = math.ceil(num_images / 10)
    #print(num_images)
    #print(num_rows)
    # タイル状画像を作成 (num_rows行10列)
    tile_images = np.full((num_rows*64, 10*64), 255, dtype=np.uint8)
    # 各画像をタイル状に配置
    for i in range(num_images):   
        row = i // 10
        col = i % 10
        tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = digit_images[i]
    # タイル状画像を描画
    showGrayscaleImage(tile_images)
    # 画像を移動する
    for x_move_pixels in Image_move_pixels:
        for y_move_pixels in Image_move_pixels:
            for i in range(num_images):
                # 画像を移動する
                target_image = np.full((64, 64), 0, dtype=np.uint8)
                x_trim = abs(x_move_pixels)
                y_trim = abs(y_move_pixels)
                source = digit_images[i][y_trim:64-y_trim, x_trim:64-x_trim]
                y_start = y_trim + y_move_pixels
                x_start = x_trim + x_move_pixels
                target_image[y_start:y_start+source.shape[0], x_start:x_start+source.shape[1]] = source

                # print(processed_image.shape)
                # 画像を保存する
                fname = os.path.join(base_dir,f'Xmove_{x_move_pixels:+03d}_Ymove_{y_move_pixels:+03d}', file_list[i])
                cv2.imwrite(fname, target_image)
                # 画像を表示する
                row = i // 10
                col = i % 10
                tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = target_image
            # タイル状画像を描画
            showGrayscaleImage(tile_images)
            # タイル状画像を保存
            tile_fname = f'../data/tile_Xmove_{x_move_pixels:+03d}_Ymove_{y_move_pixels:+03d}.png'
            cv2.imwrite(tile_fname, tile_images)

# 画像にノイズを付加する。
def addNoiseToImage (file_path, Image_addNoise_ratio, target_folder):
    image = Image.open(file_path)
    image = np.array(image)
    print(image.shape)
    print('data type',type(image))
    # 画像にノイズを付加する
    for noise_ratio in Image_addNoise_ratio:
        arr1 = np.random.rand(64, 64) 
        binary_arr = (arr1 <= noise_ratio).astype(int)
        arr2 = np.random.rand(64, 64) * 256
        noise_arr = arr2 * binary_arr
        target_image = (image + noise_arr).astype(np.uint8)
        # 画像を保存する
        fname_processed = os.path.join(target_folder, f'{os.path.basename(file_path)[:-4]}_noise_{int(noise_ratio*100):03d}' + '.png')
        cv2.imwrite(fname_processed, target_image)
        
# 指定されたフォルダにある画像にノイズを付加する。
def augmentImageByNoise (original_image_path, base_dir, Image_addNoise_ratio):
    # make directories for estimation data
    os.makedirs(base_dir, exist_ok=True)
    for noise_ratio in Image_addNoise_ratio:
        os.makedirs(os.path.join(base_dir,f'noise_{int(noise_ratio*100):03d}'), exist_ok=True)

    # original画像をloadする
    folder_path = original_image_path
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    file_list.sort()
    print(file_list)
    digit_images = []
    for fname in file_list:
        image = Image.open(os.path.join(folder_path, fname))
        image = np.array(image)
        #print(image.shape)
        #print('data type',type(image))
        digit_images.append(image)
    # 画像の数を取得
    num_images = len(file_list)
    num_rows = math.ceil(num_images / 10)
    #print(num_images)
    #print(num_rows)
    # タイル状画像を作成 (num_rows行10列)
    tile_images = np.full((num_rows*64, 10*64), 255, dtype=np.uint8)
    # 各画像をタイル状に配置
    for i in range(num_images):   
        row = i // 10
        col = i % 10
        tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = digit_images[i]
    # タイル状画像を描画
    showGrayscaleImage(tile_images)
    # 画像にノイズを付加する
    for noise_ratio in Image_addNoise_ratio:
        for i in range(num_images):
            arr1 = np.random.rand(64, 64) 
            binary_arr = (arr1 <= noise_ratio).astype(int)
            arr2 = np.random.rand(64, 64) * 256
            noise_arr = arr2 * binary_arr
            target_image = (digit_images[i] + noise_arr).astype(np.uint8)
            # 画像を保存する
            fname = os.path.join(base_dir,f'noise_{int(noise_ratio*100):03d}', file_list[i])
            cv2.imwrite(fname, target_image)
            # 画像を表示する
            row = i // 10
            col = i % 10
            tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = target_image
        # タイル状画像を描画
        showGrayscaleImage(tile_images)
        # タイル状画像を保存
        tile_fname = f'../data/tile_noise_{int(noise_ratio*100):03d}.png'
        cv2.imwrite(tile_fname, tile_images)

# 学習、検証、評価用データを作成する。元の画像を縮小し、回転する。画像のサイズはそのままで、余白は黒で埋める。
def augmentImage (original_image_path, base_dir, Image_reduction_ratio, Image_rotation_angle):
    # make directories for estimation data
    os.makedirs(base_dir, exist_ok=True)

    for ratio in Image_reduction_ratio:
        for angle in Image_rotation_angle:
            os.makedirs(os.path.join(base_dir,f'ratio_{int(ratio*100):03d}_angle_{angle:+03d}'), exist_ok=True)

    # original画像をloadする
    folder_path = original_image_path
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    file_list.sort()
    print(file_list)

    digit_images = []
    for fname in file_list:
        image = Image.open(os.path.join(folder_path, fname))
        image = np.array(image)
        #print(image.shape)
        #print('data type',type(image))
        digit_images.append(image)

    # 画像の数を取得
    num_images = len(file_list)
    num_rows = math.ceil(num_images / 10)
    #print(num_images)
    #print(num_rows)
    # タイル状画像を作成 (num_rows行10列)
    tile_images = np.full((num_rows*64, 10*64), 255, dtype=np.uint8)
    # 各画像をタイル状に配置
    for i in range(num_images):
        resized_image = resizeImage(digit_images[i], 64)
        row = i // 10
        col = i % 10
        tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = resized_image
        digit_images[i] = resized_image
    # タイル状画像を描画
    showGrayscaleImage(tile_images)

    # 画像を縮小する。回転する。
    for ratio in Image_reduction_ratio:
        for angle in Image_rotation_angle:
            for i in range(num_images):
                # 画像を縮小する
                processed_image = rotateAndresizeObject(digit_images[i], angle, ratio)
                # print(processed_image.shape)
                # 画像を保存する
                fname = os.path.join(base_dir,f'ratio_{int(ratio*100):03d}_angle_{angle:+03d}', file_list[i])
                cv2.imwrite(fname, processed_image)
                # 画像を表示する
                row = i // 10
                col = i % 10
                tile_images[row*64:(row+1)*64, col*64:(col+1)*64] = processed_image
            # タイル状画像を描画
            showGrayscaleImage(tile_images)
            # タイル状画像を保存
            tile_fname = f'../data/tile_ratio_{int(ratio*100):03d}_angle_{angle:+03d}.png'
            cv2.imwrite(tile_fname, tile_images)
# Define a convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=27, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(in_channels=27, out_channels=81, kernel_size=5, padding=0)
        # ((64/2-4)/2-4)/2 -> 5
        self.fc1 = nn.Linear(81, 40)
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        #print(x.shape)  # ここでshapeを確認
        x = torch.flatten(x, 1)  # バッチサイズを維持しつつフラット化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x
    
class CNN_GVV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, padding=0)
        # ((64/2-4)/2-4)/2 -> 5
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        #print(x.shape)  # ここでshapeを確認
        x = torch.flatten(x, 1)  # バッチサイズを維持しつつフラット化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x
    
class CNN3_GVV(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=0)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, padding=0)
        # ((64/2-4)/2-4)/2 -> 5
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        #print(x.shape)  # ここでshapeを確認
        x = torch.flatten(x, 1)  # バッチサイズを維持しつつフラット化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x
class CNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=0)
        # (64/2-4)/2 -> 14
        self.fc1 = nn.Linear(14 * 14 * 8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x
# Define a convolutional neural network
class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
        # 64/2/2 -> 16
        self.fc1 = nn.Linear(16 * 16 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, padding=2)
        # 64/2/2 -> 16
        self.fc1 = nn.Linear(16 * 16 * 8, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x   
# Define a convolutional neural network
class CNN0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # ((64-4)/2-4)/2 -> 13
        self.fc1 = nn.Linear(13 * 13 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)
        # 出力を0-1の範囲に正規化
        #x = F.softmax(x, dim=1)
        return x

class CNNGVV(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(3, 3)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.conv7(x)
        #print(x.shape)  # ここでshapeを確認
        x = torch.flatten(x, 1)  # バッチサイズを維持しつつフラット化
        #print(x.shape)  # flatten後のshapeを確認
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# カスタムデータセットの作成
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        # データとラベルを受け取るコンストラクタ
        self.data = data
        self.labels = labels

    def __len__(self):
        # データセットのサイズを返す
        return len(self.data)

    def __getitem__(self, idx):
        # 指定したインデックスのデータとラベルを返す
        # 画像をテンソルに変換
        image_tensor = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)  # チャンネル次元を追加
        sample = {'data': image_tensor, 'label': self.labels[idx]}
        return sample

# カスタムデータセットの作成（遅延読み込み）
class LazyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 画像を読み込む
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        # 画像をテンソルに変換
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # チャンネル次元を追加
        label = self.labels[idx]
        return {'data': image_tensor, 'label': label}

# datasetを作成する
def createDataset(folder_path, lazy_load=True):
    if lazy_load:
        image_paths = []
        labels = []
        folder_list = os.listdir(folder_path)
        folder_list.sort()
        for folder in folder_list:
            file_list = [f for f in os.listdir(os.path.join(folder_path, folder)) if f.endswith('.png')]
            file_list.sort()
            for fname in file_list:
                file_path = os.path.join(folder_path, folder, fname)
                image_paths.append(file_path)
                match = re.search(r'column(\d+)', fname)
                number = 0
                if match:
                    number = int(match.group(1))
                labels.append(number)
        dataset = LazyCustomDataset(image_paths, labels)
    else:
        # コア数
        print(f"physical:{psutil.cpu_count(logical=False)} logical:{psutil.cpu_count(logical=True)}")
        data = []
        labels = []
        folder_list = os.listdir(folder_path)
        folder_list.sort()
        for folder in folder_list:
            file_list = [f for f in os.listdir(os.path.join(folder_path, folder)) if f.endswith('.png')]
            file_list.sort()
            for fname in file_list:
                file_path = os.path.join(folder_path, folder, fname)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(f"Image not found: {file_path}")
                # 画像をリサイズ
                data.append(image)
                match = re.search(r'column(\d+)', fname)
                number = 0
                if match:
                    number = int(match.group(1))
                labels.append(number)            
            mem = psutil.virtual_memory()            
            print(f"used:{mem.used/1024/1024:.2f}MB total:{mem.total/1024/1024:.2f}MB {folder} done")
        dataset = CustomDataset(data, labels)
    return dataset

# networkをロードする
def loadNetwork(file_name):
    net = CNN()
    net.eval()  # 評価モードに設定
    # モデルのロード
    if os.path.exists(file_name):
        net.load_state_dict(torch.load(file_name))
        print(f"Model loaded from {file_name}")
    else:
        print(f"Model file {file_name} not found. Using uninitialized model.")
    return net

# 画像内の数字を推論する
def predict(net, img):
    with torch.no_grad():
        input_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # チャンネル次元を追加
        input_tensor = input_tensor.unsqueeze(0)  # バッチ次元を追加
        output = net(input_tensor)  # バッチ次元を追加
        predicted = torch.argmax(output, dim=1)
    return predicted.item()