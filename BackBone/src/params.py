import os

# 模型地址
DETECT_MODEL_PATH = os.path.join('model', "detect.tflite")
LANDMARK_MODEL_PATH = os.path.join('model', 'shape_predictor_68_face_landmarks.dat')
POSE_MODEL_PATH = os.path.join('model', "posenet_mobilenet.tflite")
FACE_MODEL_PATH = os.path.join('model', "arcface_mobilenet.tflite")

# 常数
THRESHOLD = 0.8
VERIFICATION_THRESHOLD = 0.95
PROB_SPOOFING = 0.1
PROB_RECOGNITION = 0.1