#!/usr/bin/env python3
"""
智能情绪感应灯 - EfficientNet-B0 WiFi控制最终版
硬件：树莓派4B + 500万摄像头 + 米家床头灯2 (MJTDo6YL)
功能：每10分钟拍照一次，识别情绪并控制灯光
"""

import time
import schedule
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from miio import Yeelight
import logging
import sys
from datetime import datetime

# ============ 配置区域 (必须修改！) ============
# 1. 米家台灯配置 (通过WiFi控制)
DEVICE_IP = "192.168.31.XXX"        # 台灯的局域网IP地址
DEVICE_TOKEN = "您的32位设备令牌"    # 台灯的Token

# 2. 模型配置
MODEL_PATH = "models/emotion_efficientnet_b0.tflite"  # TFLite模型路径
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 3. 摄像头配置
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAPTURE_FRAMES = 3  # 拍照时连续捕获几帧（取最后一帧，让摄像头稳定）

# 4. 程序配置
LOG_FILE = "emotion_light.log"  # 日志文件路径
CHECK_INTERVAL = 10  # 任务执行间隔（分钟）

# ============ 初始化日志系统 ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============ 1. TFLite情绪识别器 ============
class EfficientNetEmotionDetector:
    """使用TFLite运行EfficientNet-B0进行情绪识别"""
    
    def __init__(self, model_path):
        """初始化TFLite解释器并加载模型"""
        try:
            logger.info(f"正在加载TFLite模型: {model_path}")
            
            # 加载TFLite模型
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # 获取输入输出详情
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # 获取输入形状
            input_shape = self.input_details[0]['shape']
            self.input_height, self.input_width = input_shape[1], input_shape[2]
            
            logger.info(f"✅ 模型加载成功，输入尺寸: {self.input_width}x{self.input_height}")
            logger.info(f"模型输入详情: {self.input_details[0]}")
            logger.info(f"模型输出详情: {self.output_details[0]}")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image):
        """
        预处理图像，转换为模型需要的输入格式
        
        注意：此处的预处理必须与训练时的预处理完全一致！
        对于EfficientNet-B0，通常需要：
        1. 调整大小到224x224
        2. 应用EfficientNet特定的归一化
        """
        # 调整大小
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # 确保图像是RGB格式（如果是BGR则转换）
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            # OpenCV默认是BGR，转换为RGB
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 转换为浮点数并归一化到[0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # EfficientNet特定的预处理（ImageNet均值/标准差）
        # 这些值必须与训练时使用的值一致！
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        img_normalized[..., 0] = (img_normalized[..., 0] - mean[0]) / std[0]
        img_normalized[..., 1] = (img_normalized[..., 1] - mean[1]) / std[1]
        img_normalized[..., 2] = (img_normalized[..., 2] - mean[2]) / std[2]
        
        # 添加批次维度 [1, height, width, 3]
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_emotion(self, image):
        """
        对单张图像进行情绪识别
        
        返回: (情绪标签, 置信度)
        """
        try:
            # 预处理图像
            input_data = self.preprocess_image(image)
            
            # 设置输入张量
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # 运行推理
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 获取输出
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 解析结果
            probabilities = output_data[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            emotion = EMOTION_LABELS[predicted_class]
            
            logger.debug(f"推理时间: {inference_time:.1f}ms, 情绪: {emotion}, 置信度: {confidence:.2f}")
            
            return emotion, float(confidence)
            
        except Exception as e:
            logger.error(f"情绪识别失败: {e}")
            return "neutral", 0.0

# ============ 2. WiFi灯光控制器 ============
class WiFiLampController:
    """通过WiFi控制米家床头灯2"""
    
    def __init__(self, ip, token):
        """初始化灯光控制器"""
        self.ip = ip
        self.token = token
        self.lamp = None
        self.is_connected = False
        self._connect()
    
    def _connect(self):
        """连接到台灯"""
        try:
            logger.info(f"正在连接台灯 {self.ip}...")
            self.lamp = Yeelight(self.ip, self.token)
            
            # 测试连接
            info = self.lamp.info()
            logger.info(f"✅ 台灯连接成功！型号: {info.model}")
            self.is_connected = True
            
        except Exception as e:
            logger.error(f"❌ 台灯连接失败: {e}")
            logger.error("请检查：1. IP地址是否正确 2. Token是否正确 3. 台灯是否在线")
            self.is_connected = False
    
    def set_emotion_light(self, emotion, confidence):
        """
        根据情绪设置灯光
        
        参数:
            emotion: 情绪标签
            confidence: 置信度 (0.0-1.0)
        """
        if not self.is_connected:
            logger.warning("台灯未连接，跳过灯光设置")
            return False
        
        try:
            # 情绪到灯光参数的映射 (根据你的要求调整)
            light_config = {
                'happy':     {'brightness': 85, 'rgb': (255, 200, 100)},    # 开心
                'neutral':   {'brightness': 65, 'rgb': (220, 230, 255)},    # 平静
                'sad':       {'brightness': 45, 'rgb': (150, 180, 255)},    # 低落
                'angry':     {'brightness': 55, 'rgb': (255, 100, 100)},    # 烦躁
                'surprise':  {'brightness': 70, 'rgb': (255, 255, 150)},    # 惊讶
                'fear':      {'brightness': 40, 'rgb': (100, 100, 200)},    # 恐惧
                'disgust':   {'brightness': 50, 'rgb': (150, 200, 100)},    # 厌恶
            }
            
            # 获取配置，如果情绪未定义则使用中性光
            config = light_config.get(emotion, light_config['neutral'])
            brightness = config['brightness']
            rgb = config['rgb']
            
            # 根据置信度调整亮度（可选）
            # 如果置信度低于0.5，降低亮度变化幅度
            if confidence < 0.5:
                brightness = int(brightness * 0.7)
                logger.info(f"置信度较低({confidence:.2f})，使用柔和灯光")
            
            # 转换为设备范围 (0-255)
            device_brightness = int(brightness * 2.55)
            
            # 设置灯光
            self.lamp.set_rgb(rgb[0], rgb[1], rgb[2])
            time.sleep(0.05)  # 短暂延迟
            self.lamp.set_brightness(device_brightness)
            
            logger.info(f"💡 灯光设置: {emotion} -> 亮度{brightness}%, RGB{rgb}")
            return True
            
        except Exception as e:
            logger.error(f"设置灯光失败: {e}")
            # 尝试重新连接
            try:
                self._connect()
            except:
                logger.error("重新连接失败")
            return False

# ============ 3. 摄像头管理器 ============
class CameraManager:
    """管理摄像头的捕获和释放"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = None
    
    def capture_image(self):
        """捕获一张图像"""
        try:
            # 如果摄像头未打开，则打开
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    logger.error("无法打开摄像头")
                    return None
                
                # 设置摄像头参数
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info(f"摄像头已打开，分辨率: {self.width}x{self.height}")
            
            # 捕获多帧，让摄像头稳定（取最后一帧）
            for i in range(CAPTURE_FRAMES):
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("捕获图像失败")
                    self.release()
                    return None
            
            logger.debug(f"图像捕获成功，尺寸: {frame.shape}")
            return frame
            
        except Exception as e:
            logger.error(f"捕获图像时出错: {e}")
            return None
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.debug("摄像头资源已释放")

# ============ 4. 主任务函数 ============
def emotion_detection_task():
    """主任务：捕获图像、识别情绪、控制灯光"""
    logger.info("=" * 50)
    logger.info(f"开始情绪识别任务 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化摄像头
    camera = CameraManager(CAMERA_WIDTH, CAMERA_HEIGHT)
    
    # 捕获图像
    frame = camera.capture_image()
    if frame is None:
        logger.error("图像捕获失败，跳过本次任务")
        camera.release()
        return
    
    # 可选：保存图像用于调试
    # cv2.imwrite(f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", frame)
    
    # 释放摄像头（重要：长时间占用摄像头可能有问题）
    camera.release()
    
    # 检测人脸（可选，提高准确性）
    # 如果没有人脸，可以跳过情绪识别
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        logger.warning("未检测到人脸，跳过情绪识别")
        # 可以设置默认灯光或保持当前状态
        return
    
    # 识别情绪
    emotion, confidence = detector.predict_emotion(frame)
    
    # 记录结果
    logger.info(f"识别结果: {emotion} (置信度: {confidence:.2%})")
    
    # 控制灯光
    if confidence > 0.3:  # 置信度阈值
        lamp_controller.set_emotion_light(emotion, confidence)
    else:
        logger.warning(f"置信度过低({confidence:.2%})，不调整灯光")
    
    logger.info(f"任务完成，耗时: {time.time() - task_start_time:.1f}秒")
    logger.info("=" * 50)

# ============ 5. 程序入口点 ============
if __name__ == "__main__":
    print("=" * 60)
    print("智能情绪感应灯 - EfficientNet-B0 WiFi控制版")
    print(f"识别频率: 每{CHECK_INTERVAL}分钟一次")
    print("=" * 60)
    
    # 全局变量
    detector = None
    lamp_controller = None
    task_start_time = 0
    
    try:
        # 初始化情绪检测器
        logger.info("初始化情绪检测器...")
        detector = EfficientNetEmotionDetector(MODEL_PATH)
        
        # 初始化灯光控制器
        logger.info("初始化灯光控制器...")
        lamp_controller = WiFiLampController(DEVICE_IP, DEVICE_TOKEN)
        
        # 设置定时任务
        logger.info(f"设置定时任务，每{CHECK_INTERVAL}分钟执行一次...")
        schedule.every(CHECK_INTERVAL).minutes.do(
            lambda: emotion_detection_task()
        )
        
        # 立即执行一次初始任务
        logger.info("执行初始情绪识别...")
        task_start_time = time.time()
        emotion_detection_task()
        
        logger.info("定时任务已启动，进入主循环...")
        logger.info("按 Ctrl+C 退出程序")
        print("\n程序运行中...")
        print("情绪识别日志将显示在上方并保存到日志文件")
        print("-" * 60)
        
        # 主循环
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次任务
            
    except KeyboardInterrupt:
        logger.info("收到退出信号，正在清理...")
    except Exception as e:
        logger.error(f"程序运行错误: {e}", exc_info=True)
    finally:
        # 清理工作
        logger.info("程序结束")
        
        # 设置一个柔和的默认灯光
        if lamp_controller and lamp_controller.is_connected:
            try:
                lamp_controller.lamp.set_rgb(255, 220, 180)
                lamp_controller.lamp.set_brightness(76)  # 30%亮度
                logger.info("已设置柔和默认灯光")
            except:
                pass
