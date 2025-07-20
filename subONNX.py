import onnxruntime as ort
import numpy as np

class Onnx:
    def __init__(self, file_name):
        # ONNXモデルを読み込む
        self.ort_session = ort.InferenceSession(file_name)
        self.ort_session.disable_fallback()
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        # 入力層の名前を取得
        print("input:")
        for session_input in self.ort_session.get_inputs():
            print(session_input.name, session_input.shape)
        # 出力層の名前を取得
        print("output:")
        for session_output in self.ort_session.get_outputs():
            print(session_output.name, session_output.shape)
        
    def predict(self, image):
        input_onnx = image.astype(np.float32)
        input_onnx = np.expand_dims(input_onnx, axis=0)  # (1, H, W)
        input_onnx = np.expand_dims(input_onnx, axis=0)  # (1, 1, H, W)
        out_onnx = self.ort_session.run([self.output_name], {self.input_name: input_onnx})
        out = out_onnx[0][0]
        #print(out)
        # 確率を昇順にソートする
        max_value = np.max(out)
        max_idx = np.argmax(out)
        return max_idx, max_value