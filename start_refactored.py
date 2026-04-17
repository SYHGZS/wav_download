"""
重构后的 start.py 文件
将原来的 start_detection 方法拆分为两个独立的方法：
1. capture_frames_from_cameras: 同步截图操作
2. process_detection_on_captured_frames: 对截图数据进行识别
"""

import os
import random
import time

import cv2
import numpy as np
import pymysql
import requests
import torch
from dbutils.persistent_db import PersistentDB
from ultralytics import YOLO

from cos_util import upload_to_cos
from video_capture_utils import capture_frame_robust


def _ts():
    """返回当前时间字符串，用于日志前缀"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_detection_interval() -> int:
    """
    根据当前时段返回检测间隔（秒）：
      高峰时段（11:00–14:00 / 17:00–20:00）：每 2 分钟
      其他时段：每 10 分钟
    """
    hour = time.localtime().tm_hour
    if 11 <= hour < 14 or 17 <= hour < 20:
        return 120  # 2 分钟
    return 600  # 10 分钟


def is_invalid_frame(frame, std_threshold=15.0, black_ratio_threshold=0.80, check_size=(320, 240)):
    """
    判断截图是否为无效帧（全灰、全黑、摄像头离线等）。
    - 条件1：灰度图像素标准差极低 → 全灰/纯色帧（摄像头离线/信号中断）
    - 条件2：近黑像素占比超过阈值 → 全黑帧（遮挡/断电），与 shelter.py 逻辑一致
    任一条件满足即判定为无效帧，返回 (True, 原因说明)；否则返回 (False, None)。
    """
    # 1. 防御性检查：确保 frame 是有效的 NumPy 数组且具有完整的三通道
    if not isinstance(frame, np.ndarray):
        return True, "Invalid type: frame is not numpy array"

    if frame.size == 0 or len(frame.shape) != 3 or frame.shape[2] != 3:
        return True, f"Invalid shape: {frame.shape}"

    height, width = frame.shape[:2]
    if height == 0 or width == 0:
        return True, "Invalid dimensions: width or height is 0"

    # 2. 性能优化：缩小图像后再进行统计计算
    # 缩小图像能保留宏观的颜色分布，同时极大降低 CPU 计算开销
    small_frame = cv2.resize(frame, check_size, interpolation=cv2.INTER_NEAREST)

    # 3. 标准差检测（纯色帧/离线帧）
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    if std_dev < std_threshold:
        return True, f"std={std_dev:.1f} < {std_threshold}"

    # 4. 黑屏检测（遮挡/断电）
    # 在 NumPy 中直接计算，small_frame 已经足够小，计算极快
    black_mask = np.all(small_frame <= 30, axis=2)
    black_ratio = float(np.sum(black_mask)) / (check_size[0] * check_size[1])

    if black_ratio > black_ratio_threshold:
        return True, f"black_ratio={black_ratio:.2%} > {black_ratio_threshold:.0%}"

    return False, None


# 配置信息（保持不变）
headers = {
    "Client-Id": "cycm_jg_pc_110107",
    "Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1c2VySWQiOiJ2dkoxamU4VTJiT0wyQXExUTloTXRHS2IvNzFxcUNKbkczVElqNTRYbEZzak9pTEcwdWpWVmRpVTd3c25DYWhrdXBOVk5nWGhiaGs3dTJiTlpxajhWSEdWRjY5UW5HWXZXWVBoWEdqSHZFNko3bTFqenJJOC9CTzg1WVhrSnhSNkY4UU5VQU5UY0xEeVFkOXRZUUcrd1NiNzBBZ0JlS3Y0TEpuTXVUUEJsQkNIbFNuU0xiOUpBTWUyU1BaVEV0NXV0aGIxUEY1YWRjZXFlekJZajRFNU5GWTNEd20rQUF2WTREOHFqZ2ZhUjJXVFJGVXdOY2ZnQnk0Q0RHNUd2Mng2WDJPN2xQVTdpYUlzZHFvVEU4RnVLRy9HSkhyWXNkWEg3OVo0WlVUdlVKanRKa2xnMGk3c2xtWEFNaTV2NGRPSUExUG9HM1cyL0pkNXJISytxR0xETXc9PSIsImlhdCI6MTc3NjM0NDMxMiwiZXhwIjoyNDQ4MjI2NDMxMn0.NN20PvRRnTUwXAckBd9PMlipkXtx19kCBx95UchPjz2ivqYTznre6gGnqPkS-Rp2AkDUkArEamU-1SIItsVvvXbledeciFmK_SVbsYGTgb2-zwLn6-05fbAuxpMQz-yV6DQKRd2BYwFRaeG9m3_7xXqDpmbvu6lkIPItT20bTPzIAqvjIa2SSgrJcuvxZkLMDdE11m_V8AeJpSqFWJpMXaySVz4GuIQl2gphvMJv0vwBhssLYsORbQBTAY8p8qE-_fFDHuTB4W-Lxxkj2iaYpRSGm5ym5IGMo7M-gf5VyzI68kWcnwauO8SdvffS-mpu0ex_8w7uP_ogPKgXQMPMJA",
}


# ============================================================
# 人离火检测参数
# ============================================================
STOVE_IOA_THRESHOLD = 0.3  # 明火/蒸汽与锅的交占比阈值
STOVE_ATTEND_DISTANCE_THRESHOLD = 1.5  # 归一化距离阈值（< 该值视为有人看管）
STOVE_CLS_FIRE_THRESHOLD = 0.4  # 明火分类器置信度阈值
STOVE_CLS_BOIL_THRESHOLD = 0.4  # 沸腾分类器置信度阈值
STOVE_CLS_STEAM_THRESHOLD = 0.4  # 蒸汽分类器置信度阈值
STOVE_STEAM_EXTEND_RATIO = 0.5  # 蒸汽裁图向上扩展比例

# ============================================================
# 赤膊检测参数
# ============================================================
BARENESS_BODY_CONF = 0.6  # 人体检测置信度阈值
BARENESS_CLS_CONF = 0.75  # 赤膊分类器置信度阈值


# ============================================================
# 人离火辅助函数（独立，便于后续升级为多帧版）
# ============================================================


def calculate_ioa(small_box, large_box):
    """计算交占比 IoA = intersection / small_box_area"""
    xA = max(small_box[0], large_box[0])
    yA = max(small_box[1], large_box[1])
    xB = min(small_box[2], large_box[2])
    yB = min(small_box[3], large_box[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
    return inter_area / float(small_area + 1e-5)


def safe_crop(frame, bbox):
    """安全裁剪，坐标越界或区域为空时返回 None"""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def classify_crop_positive(model, crop, threshold):
    """对裁剪图像做分类，class index 1 为正样本，返回 (is_positive, score)"""
    result = model(crop, verbose=False)[0]
    class_index = result.probs.top1
    score = float(result.probs.top1conf)
    return class_index == 1 and score >= threshold, score


def get_half_bbox(pot_bbox, mode, img_h, img_w):
    """
    获取锅 bbox 的裁剪区域：
      mode='fire'  — 下半区（用于明火分类）
      mode='boil'  — 上半区（用于沸腾分类）
      mode='steam' — 上半区 + 向上扩展（用于蒸汽分类）
    """
    x1, y1, x2, y2 = pot_bbox
    mid_y = (y1 + y2) / 2
    if mode == "fire":
        return [max(0, int(x1)), max(0, int(mid_y)), min(img_w, int(x2)), min(img_h, int(y2))]
    elif mode == "boil":
        return [max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(mid_y))]
    elif mode == "steam":
        height = y2 - y1
        extend = height * STOVE_STEAM_EXTEND_RATIO
        return [max(0, int(x1)), max(0, int(y1 - extend)), min(img_w, int(x2)), min(img_h, int(mid_y))]
    return [int(x1), int(y1), int(x2), int(y2)]


def check_pot_attended_by_pose(pot_bbox, pose_results, threshold=STOVE_ATTEND_DISTANCE_THRESHOLD):
    """
    基于姿态关键点判断锅具是否有人看管。
    计算所有人的肩(5,6)、肘(7,8)、腕(9,10)到锅中心的最短归一化距离，小于 threshold 即视为有人看管。
    """
    pot_cx = (pot_bbox[0] + pot_bbox[2]) / 2
    pot_cy = (pot_bbox[1] + pot_bbox[3]) / 2
    l_pot = ((pot_bbox[2] - pot_bbox[0]) ** 2 + (pot_bbox[3] - pot_bbox[1]) ** 2) ** 0.5
    min_dist = float("inf")
    if pose_results is not None and pose_results.keypoints is not None:
        for kps in pose_results.keypoints.xy.cpu().numpy():
            for idx in [5, 6, 7, 8, 9, 10]:  # 肩、肘、腕
                if len(kps) > idx and kps[idx][0] != 0:
                    dist = ((kps[idx][0] - pot_cx) ** 2 + (kps[idx][1] - pot_cy) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
    d_norm = min_dist / (l_pot + 1e-5)
    return d_norm < threshold


# ============================================================
# 赤膊辅助函数
# ============================================================


def extract_upper_body(image, box, upper_ratio=0.45):
    """从人体检测框中裁取上半身区域（默认取 bbox 上方 45% 高度）"""
    x1, y1, x2, y2 = map(int, box)
    height = y2 - y1
    upper_y2 = min(y1 + int(height * upper_ratio), image.shape[0])
    crop = image[y1:upper_y2, x1:x2]
    return crop if crop.size > 0 else None


class DetectionSystem:
    def __init__(self, environment=None):
        """初始化检测系统，设置环境相关配置"""
        # 根据传入的environment参数设置host
        if environment == "test":
            self.host = "https://cycm.jyfwyun.com/apiv3/"
            print("使用测试环境host:", self.host)
        else:
            self.host = "https://cycm.jyfwyun.com/apiv3/"
            print("使用生产环境host:", self.host)

        # 设置数据库连接参数
        if environment == "test":
            db_config = {
                "host": "36.112.39.92",
                "port": 61851,
                "user": "pgy-cy",
                "password": "JVOjAiMBAUXOlVV6",
                "db": "pgy_server",
            }
            print("使用测试环境数据库:", db_config["host"])
        else:
            db_config = {
                "host": "pgy-cy.prod.mysql",
                "port": 51851,
                "user": "pgy-cy",
                "password": "JVOjAiMBAUXOlVV6",
                "db": "pgy_server",
            }
            print("使用生产环境数据库:", db_config["host"])

        self.pool = PersistentDB(pymysql, 5, **db_config)
        print(f"[{_ts()}] [INIT] 数据库连接池创建成功: host={db_config['host']}, port={db_config['port']}")

        # --- 二级检测模型配置 ---
        self.MODEL_GROUPS = [
            {
                "group_name": "garbage_bin_lid_group",  # 模型组名称
                "detector_path": "./garbage_bin_detection_with_tuning_v1.0.pt",  # 检测模型路径
                "classifier_path": "./kitchen_classification_garbage_bin_v1.0.pt",  # 分类模型路径
                "yolo_target_classes": ["bin"],  # YOLO检测模型要关注的类别
                "classifier_class_names": [
                    "close lid",
                    "not close lid",
                    "others",
                ],  # 分类模型的所有类别标签 (请务必按训练时的顺序填写)
                "display_classes": ["not close lid"],  # 我们最终希望上报的违规分类结果
                "detection_type": "not_close_lid",  # 自定义一个唯一的违规类型名称，用于上报
            },
            {
                "group_name": "person_work_clothes_group",
                "detector_path": "./person_detection_with_tuning_v1.1.pt",
                "classifier_path": "./kitchen_classification_person_v1.0.pt",
                "yolo_target_classes": ["person"],
                "classifier_class_names": ["others", "with_work_clothes", "without_work_clothes"],
                "display_classes": ["without_work_clothes"],
                "detection_type": "without_work_clothes",  # 自定义一个唯一的违规类型名称
            },
        ]

        print(f"[{_ts()}] [INIT] 开始加载模型缓存...")
        _model_defs = {
            "smoke": "./yolo11_smoke.pt",
            "phone": "./yolo11_phone.pt",
            "kitchen": "./yolo11_kitchen.pt",
            "bareness_cls": "./yolo_shirtless_cls_v2.pt",
            "stove_pot": "./yolo11_pot.pt",
            "stove_gas": "./steam_detection_v1.0.pt",
            "stove_pose": "./yolo11s-pose.pt",
            "stove_cls_fire": "./yolo11_fire_class.pt",
            "stove_cls_boil": "./yolo11_boil_class.pt",
            "stove_cls_steam": "./yolo11_steam_class.pt",
        }
        self.model_cache = {}
        for _name, _path in _model_defs.items():
            try:
                self.model_cache[_name] = YOLO(_path)
                print(f"[{_ts()}] [INIT] ✓ 模型加载成功: {_name} ({_path})")
            except Exception as e:
                print(f"[{_ts()}] [INIT] ✗ 模型加载失败: {_name} ({_path}) - {e}")

        # --- 执行模型加载 ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{_ts()}] [INIT] 所有模型缓存加载完成，使用设备: {self.DEVICE}")
        self.two_stage_models = self.load_two_stage_models(self.DEVICE)

        # 存储截图数据的全局变量
        self.captured_frames = {}

    # --- 二级模型加载函数和缓存 ---
    def load_two_stage_models(self, device):
        """根据MODEL_GROUPS配置加载所有二级模型到指定设备"""
        loaded_models = {}
        print("\n--- Loading Two-Stage Models ---")
        for group in self.MODEL_GROUPS:
            group_name = group["group_name"]
            print(f"--- Loading model group: {group_name} ---")
            try:
                detector = YOLO(group["detector_path"])
                detector.to(device)
                print(f"  Detector '{group['detector_path']}' loaded.")

                classifier = YOLO(group["classifier_path"])
                classifier.to(device)
                print(f"  Classifier '{group['classifier_path']}' loaded.")

                loaded_models[group_name] = {"detector": detector, "classifier": classifier, "config": group}
            except Exception as e:
                print(f"Error loading model group {group_name}: {e}")

        print("--- All Two-Stage Models Loaded ---\n")
        return loaded_models

    # 方法一：同步截图操作
    def capture_frames_from_cameras(self):
        """
        对所有摄像头进行同步截图操作，保存到本地并存储到对象中
        """
        conn = self.pool.connection()
        cs = conn.cursor()

        # 查询符合条件的摄像头
        sql = "SELECT id,shop_id,detection_models,camera_name,device_serial FROM pgy_server.pgy_shop_camera where enable = 1 and length(shop_id) > 1"
        cs.execute(sql)
        camera_list = cs.fetchall()
        print(f"[{_ts()}] [CAPTURE] 开始截图任务，共 {len(camera_list)} 个启用摄像头")
        success_count = 0
        fail_count = 0

        def capture_single_camera(camera):
            """单个摄像头截图的函数"""
            nonlocal success_count, fail_count
            camera_id = camera[0]
            shop_id = camera[1]
            detection_models = camera[2]
            device_serial = camera[4]
            camera_name = camera[3] or camera_id  # 摄像头名称，无则用 ID 代替
            try:
                rtsp = None
                # 调用接口获取RTSP直播地址
                response = requests.post(
                    self.host + "cycm-cloud/cross/camera/getVideoUrl?id=" + camera_id, headers=headers
                )
                if response.status_code == 200 and response.json()["code"] == 0:
                    result = response.json()["result"]
                    result["rtsp"] = (
                        f"rtsp://wvp2.theling.team:8554/rtp/{device_serial}?originTypeStr=rtp_push&videoCodec=H265"
                    )
                    # 优先使用RTSP流，如果没有则使用HLS流
                    if "rtsp" in result and result["rtsp"]:
                        rtsp = result["rtsp"]
                        print(f"[{_ts()}] [CAPTURE] 摄像头 [{camera_name}](shop={shop_id}) 获取视频流成功 (RTSP)")
                    elif "hls" in result and result["hls"]:
                        rtsp = result["hls"]
                        print(f"[{_ts()}] [CAPTURE] 摄像头 [{camera_name}](shop={shop_id}) 获取视频流成功 (HLS)")
                    else:
                        print(
                            f"[{_ts()}] [CAPTURE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 接口未返回可用视频流 (无RTSP/HLS)"
                        )
                        fail_count += 1
                        return
                else:
                    print(
                        f"[{_ts()}] [CAPTURE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 获取视频流失败: HTTP状态={response.status_code}"
                    )
                    fail_count += 1
                    return

                if rtsp is None:
                    print(f"[{_ts()}] [CAPTURE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 视频流地址为空")
                    fail_count += 1
                    return
                # 截图
                frame = capture_frame_robust(rtsp)
                if frame is not None:
                    invalid, reason = is_invalid_frame(frame)
                    if invalid:
                        print(
                            f"[{_ts()}] [CAPTURE SKIP] 摄像头 [{camera_name}](shop={shop_id}) 无效帧，已跳过: {reason}"
                        )
                        fail_count += 1
                        return
                    file_name = str(int(time.time())) + str(random.randint(0, 50)) + ".jpg"
                    # device_serial 结构：shopid_cameraid
                    if "_" in device_serial:
                        shop_id_from_serial, camera_id_from_serial = device_serial.split("_", 1)
                    else:
                        shop_id_from_serial, camera_id_from_serial = shop_id, device_serial
                    # 在img下创建年月日/店铺ID目录
                    img_dir = os.path.join("./img", time.strftime("%Y%m%d", time.localtime()), shop_id_from_serial)
                    os.makedirs(img_dir, exist_ok=True)
                    file_path = os.path.join(img_dir, file_name)
                    cv2.imwrite(file_path, frame)
                    h, w = frame.shape[:2]
                    # 存储截图信息到全局对象中（不再需要线程锁）
                    self.captured_frames[camera_id] = {
                        "file_path": file_path,
                        "camera_id": camera_id,
                        "shop_id": shop_id,
                        "camera_name": camera_name,
                        "detection_models": detection_models,
                        "original_frame": frame,  # 保留内存中的原始帧，供检测方法直接使用
                    }
                    success_count += 1
                    print(
                        f"[{_ts()}] [CAPTURE OK] 摄像头 [{camera_name}](shop={shop_id}) 截图成功: {file_path} ({w}x{h})"
                    )
                else:
                    print(
                        f"[{_ts()}] [CAPTURE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 截图失败 (capture_frame_robust 返回 None)"
                    )
                    fail_count += 1
            except Exception as e:
                print(
                    f"[{_ts()}] [CAPTURE ERROR] 摄像头 [{camera_name}](shop={shop_id}) 截图异常: {type(e).__name__} - {e}"
                )
                fail_count += 1

        # 同步执行截图操作
        for camera in camera_list:
            capture_single_camera(camera)

        cs.close()
        conn.close()
        print(f"[{_ts()}] [CAPTURE END] 截图任务完成 - 成功 {success_count}/{len(camera_list)}，失败 {fail_count}")

    # ---  二级检测流程函数 ---
    def process_two_stage_detection(self, frame_data, camera_id, group_name, conf):
        """
        对单张截图执行新的二级（检测+分类）流程，并复用原有的结果上报逻辑。
        """
        camera_name = frame_data.get("camera_name", camera_id)
        shop_id = frame_data.get("shop_id", "?")
        print(f"[{_ts()}] [2STAGE] 摄像头 [{camera_name}](shop={shop_id}) 开始二级检测 [{group_name}]")
        original_image = frame_data["original_frame"]  # 直接使用内存中的图像，避免重复读取
        if original_image is None:
            print(f"[{_ts()}] [2STAGE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 无法获取原始图像帧")
            return

        # 遍历所有配置好的二级模型组
        models = self.two_stage_models[group_name]
        detector = models["detector"]
        classifier = models["classifier"]
        config = models["config"]

        yolo_class_names = detector.names
        yolo_target_ids = [k for k, v in yolo_class_names.items() if v in config["yolo_target_classes"]]

        # 1. 执行第一级检测
        detections = detector(original_image, classes=yolo_target_ids, conf=conf, verbose=False)
        det_count = len(detections[0].boxes) if detections[0].boxes is not None else 0
        print(
            f"[{_ts()}] [2STAGE] 摄像头 {camera_id} 第一级检测完成，检测到 {det_count} 个 {config['yolo_target_classes']} 目标"
        )
        if det_count == 0:
            return

        # 2. 遍历检测到的每个物体
        for box in detections[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # 裁剪目标
            cropped_object_cv2 = original_image[y1:y2, x1:x2]
            if cropped_object_cv2.size == 0:
                continue

            # 3. 执行第二级分类
            output = classifier(cropped_object_cv2, verbose=False)
            result = output[0]
            probs = result.probs
            class_index = probs.top1
            class_confidence = probs.top1conf.item()
            predicted_class_name = config["classifier_class_names"][class_index]

            print(
                f"[{_ts()}] [2STAGE] 摄像头 {camera_id} 分类结果: '{predicted_class_name}' (conf={class_confidence:.2f})"
            )

            # 4. 判断是否为需要上报的违规结果
            if predicted_class_name in config["display_classes"] and class_confidence > 0.6:
                print(
                    f"[{_ts()}] [VIOLATION] ★ 二级检测违规: '{predicted_class_name}' 摄像头={camera_id} conf={class_confidence:.2f}"
                )
                # === 5. 复用原项目的结果处理和上报逻辑 ===
                try:
                    # a. 绘制结果并保存图片
                    label_text = f"{predicted_class_name}: {class_confidence:.2f}"
                    result_image = original_image.copy()
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(result_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                    file_name = f"{time_str}_{config['detection_type']}_{random.randint(0, 999)}.png"
                    # 保存路径加上shop_id
                    result_dir = os.path.join("./result", time.strftime("%Y%m%d", time.localtime()), str(shop_id))
                    os.makedirs(result_dir, exist_ok=True)
                    result_file_path = os.path.join(result_dir, file_name)
                    cv2.imwrite(result_file_path, result_image)

                    # b. 上传文件到云存储
                    file_url = f"{self.host}/cloud-service/api/file/upload/buildKeysVo"
                    file_obj = {
                        "files": [{"name": file_name, "busiScene": "19_31", "fileName": file_name, "permission": 2}]
                    }
                    response = requests.post(file_url, json=file_obj, headers=headers)
                    file_data = response.json()["result"]["files"]
                    cos_key = file_data[0]["cosKey"]
                    zlFileId = file_data[0]["zlFileId"]
                    upload_to_cos(result_file_path, cos_key)
                    print(f"[{_ts()}] [UPLOAD OK] 二级检测图片上传成功: {file_name}")

                    # c. 创建检测记录
                    detection_result_url = f"{self.host}/cycm-cloud/api/detectionResult/insert"
                    detection_result = {
                        "label": config["detection_type"],  # 使用配置中定义的唯一类型名
                        "dataId": frame_data["shop_id"],
                        "detectionType": config["detection_type"],
                        "eventType": config["detection_type"],
                        "score": class_confidence,
                        "cameraId": camera_id,
                        "fileId": zlFileId,
                    }
                    response = requests.post(detection_result_url, json=detection_result, headers=headers)

                    # d. 文件关联
                    if response.json().get("result"):
                        resource_data = {"relId": response.json()["result"], "fileList": [zlFileId]}
                        file_relevance_resource_url = f"{self.host}/cloud-service/api/file/upload/fileRelevanceResource"
                        requests.post(file_relevance_resource_url, json=resource_data, headers=headers)

                    print(
                        f"[{_ts()}] [2STAGE DONE] 摄像头 [{camera_name}](shop={shop_id}) 违规记录已保存: {label_text}"
                    )

                except Exception as e:
                    print(
                        f"[{_ts()}] [2STAGE ERROR] 摄像头 [{camera_name}](shop={shop_id}) 处理二级检测结果异常: {type(e).__name__} - {e}"
                    )

    # --- 赤膊检测方法 ---
    def process_bareness_detection(self, frame_data, camera_id, conf):
        """
        赤膊检测（单帧版）：
          1. 使用人体检测器找出画面中所有人
          2. 裁取每人的上半身区域
          3. 用赤膊分类器判断是否违规（class 1 = 赤膊）
          4. 违规则绘图上报
        当前为单帧版，后续可替换为多帧投票版以降低误报率。
        """
        camera_name = frame_data.get("camera_name", camera_id)
        shop_id = frame_data.get("shop_id", "?")
        print(f"[{_ts()}] [BARENESS] 摄像头 [{camera_name}](shop={shop_id}) 开始赤膊检测")
        original_image = frame_data.get("original_frame")
        if original_image is None:
            print(f"[{_ts()}] [BARENESS FAIL] 摄像头 [{camera_name}](shop={shop_id}) 无法获取原始图像帧")
            return

        # 复用工作服检测组的人体检测器（person_detection_with_tuning_v1.1.pt）
        body_detector = self.two_stage_models["person_work_clothes_group"]["detector"]
        bareness_classifier = self.model_cache["bareness_cls"]

        detections = body_detector(original_image, conf=BARENESS_BODY_CONF, verbose=False)
        person_count = len(detections[0].boxes) if detections[0].boxes is not None else 0
        print(f"[{_ts()}] [BARENESS] 摄像头 [{camera_name}](shop={shop_id}) 人体检测完成，检测到 {person_count} 个人体")
        if person_count == 0:
            return
        for box in detections[0].boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            upper_body = extract_upper_body(original_image, coords)
            if upper_body is None:
                continue

            output = bareness_classifier(upper_body, verbose=False)[0]
            class_index = output.probs.top1
            score = float(output.probs.top1conf)
            is_violation = class_index == 1 and score >= BARENESS_CLS_CONF
            print(
                f"[{_ts()}] [BARENESS] 摄像头 {camera_id} 赤膊分类: class={class_index} score={score:.2f} violation={is_violation}"
            )

            if not is_violation:
                continue

            print(f"[{_ts()}] [VIOLATION] ★ 发现赤膊违规 摄像头=[{camera_name}](shop={shop_id}) conf={score:.2f}")
            try:
                x1, y1, x2, y2 = coords
                label_text = f"bareness: {score:.2f}"
                result_image = original_image.copy()
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(result_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

                time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                file_name = f"{time_str}_bareness_{random.randint(0, 999)}.png"
                result_dir = "./result/" + time.strftime("%Y%m%d", time.localtime()) + "/"
                os.makedirs(result_dir, exist_ok=True)
                result_file_path = os.path.join(result_dir, file_name)
                cv2.imwrite(result_file_path, result_image)

                file_url = f"{self.host}/cloud-service/api/file/upload/buildKeysVo"
                file_obj = {
                    "files": [{"name": file_name, "busiScene": "19_31", "fileName": file_name, "permission": 2}]
                }
                response = requests.post(file_url, json=file_obj, headers=headers)
                file_data = response.json()["result"]["files"]
                cos_key = file_data[0]["cosKey"]
                zlFileId = file_data[0]["zlFileId"]
                upload_to_cos(result_file_path, cos_key)
                print(f"[{_ts()}] [UPLOAD OK] 摄像头 [{camera_name}](shop={shop_id}) 赤膊违规图片上传成功: {file_name}")

                detection_result_url = f"{self.host}/cycm-cloud/api/detectionResult/insert"
                detection_result = {
                    "label": "bareness",
                    "dataId": frame_data["shop_id"],
                    "detectionType": "bareness",
                    "eventType": "bareness",
                    "score": score,
                    "cameraId": camera_id,
                    "fileId": zlFileId,
                }
                response = requests.post(detection_result_url, json=detection_result, headers=headers)

                if response.json().get("result"):
                    resource_data = {"relId": response.json()["result"], "fileList": [zlFileId]}
                    requests.post(
                        f"{self.host}/cloud-service/api/file/upload/fileRelevanceResource",
                        json=resource_data,
                        headers=headers,
                    )

                print(f"[{_ts()}] [BARENESS DONE] 摄像头 [{camera_name}](shop={shop_id}) 赤膊违规已上报: {label_text}")
            except Exception as e:
                print(
                    f"[{_ts()}] [BARENESS ERROR] 摄像头 [{camera_name}](shop={shop_id}) 处理赤膊结果异常: {type(e).__name__} - {e}"
                )

    # --- 人离火检测方法 ---
    def process_stove_detection(self, frame_data, camera_id, conf):
        """
        人离火检测（单帧版）：
          1. 检测画面中的锅具
          2. 对每口锅分别进行明火/沸腾/蒸汽分类，判断是否处于加热状态
          3. 若锅在加热，使用姿态估计判断是否有人看管
          4. 加热且无人看管则上报
        当前为单帧版，不含漏桶积分和持久化追踪，后续可升级为多帧状态版。
        """
        camera_name = frame_data.get("camera_name", camera_id)
        shop_id = frame_data.get("shop_id", "?")
        print(f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 开始人离火检测")
        original_image = frame_data.get("original_frame")
        if original_image is None:
            print(f"[{_ts()}] [STOVE FAIL] 摄像头 [{camera_name}](shop={shop_id}) 无法获取原始图像帧")
            return

        img_h, img_w = original_image.shape[:2]
        pot_model = self.model_cache["stove_pot"]
        gas_model = self.model_cache["stove_gas"]
        pose_model = self.model_cache["stove_pose"]
        cls_fire = self.model_cache["stove_cls_fire"]
        cls_boil = self.model_cache["stove_cls_boil"]
        cls_steam = self.model_cache["stove_cls_steam"]

        # 1. 检测锅具
        pot_results = pot_model(original_image, conf=conf, verbose=False)[0]
        if pot_results.boxes is None or len(pot_results.boxes) == 0:
            print(f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 未检测到锅具")
            return
        print(f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 检测到 {len(pot_results.boxes)} 口锅具")

        # 2. 全图蒸汽检测（作为兜底证据）
        gas_results = gas_model(original_image, conf=0.35, verbose=False)[0]
        gas_bboxes = gas_results.boxes.xyxy.cpu().numpy() if gas_results.boxes is not None else []
        print(
            f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 全图蒸汽/气体检测完成，检测到 {len(gas_bboxes)} 处"
        )

        # 3. 姿态检测（一次性对全图，供所有锅具复用）
        pose_results = pose_model(original_image, conf=0.5, verbose=False)[0]
        pose_count = len(pose_results.keypoints.xy) if pose_results.keypoints is not None else 0
        print(f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 姿态检测完成，检测到 {pose_count} 个人体")

        for box in pot_results.boxes:
            pot_bbox = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = pot_bbox.astype(int)

            # 4. 分类判断是否处于加热状态
            has_flame, has_boil, has_steam, has_gas = False, False, False, False

            fire_crop = safe_crop(original_image, get_half_bbox(pot_bbox, "fire", img_h, img_w))
            if fire_crop is not None:
                has_flame, _ = classify_crop_positive(cls_fire, fire_crop, STOVE_CLS_FIRE_THRESHOLD)

            boil_crop = safe_crop(original_image, get_half_bbox(pot_bbox, "boil", img_h, img_w))
            if boil_crop is not None:
                has_boil, _ = classify_crop_positive(cls_boil, boil_crop, STOVE_CLS_BOIL_THRESHOLD)

            steam_crop = safe_crop(original_image, get_half_bbox(pot_bbox, "steam", img_h, img_w))
            if steam_crop is not None:
                has_steam, _ = classify_crop_positive(cls_steam, steam_crop, STOVE_CLS_STEAM_THRESHOLD)

            for gb in gas_bboxes:
                if calculate_ioa(gb, pot_bbox) > STOVE_IOA_THRESHOLD:
                    has_gas = True
                    break

            is_active = has_flame or has_boil or has_steam or has_gas
            print(
                f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 锅具 [{x1},{y1},{x2},{y2}]: "
                f"flame={has_flame} boil={has_boil} steam={has_steam} gas={has_gas} active={is_active}"
            )

            if not is_active:
                continue

            # 5. 判断是否有人看管
            is_attended = check_pot_attended_by_pose(pot_bbox, pose_results)
            print(
                f"[{_ts()}] [STOVE] 摄像头 [{camera_name}](shop={shop_id}) 锅具 [{x1},{y1},{x2},{y2}] 看管状态: attended={is_attended}"
            )

            if is_attended:
                continue

            # 6. 上报违规
            print(
                f"[{_ts()}] [VIOLATION] ★ 发现人离火违规! 摄像头=[{camera_name}](shop={shop_id}) 锅具=[{x1},{y1},{x2},{y2}]"
            )
            try:
                active_flags = []
                if has_flame:
                    active_flags.append("F")
                if has_boil:
                    active_flags.append("B")
                if has_steam or has_gas:
                    active_flags.append("G")
                label_text = f"stove_unattended [{''.join(active_flags) or '-'}]"

                result_image = original_image.copy()
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(result_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                file_name = f"{time_str}_stove_unattended_{random.randint(0, 999)}.png"
                result_dir = "./result/" + time.strftime("%Y%m%d", time.localtime()) + "/"
                os.makedirs(result_dir, exist_ok=True)
                result_file_path = os.path.join(result_dir, file_name)
                cv2.imwrite(result_file_path, result_image)

                file_url = f"{self.host}/cloud-service/api/file/upload/buildKeysVo"
                file_obj = {
                    "files": [{"name": file_name, "busiScene": "19_31", "fileName": file_name, "permission": 2}]
                }
                response = requests.post(file_url, json=file_obj, headers=headers)
                file_data = response.json()["result"]["files"]
                cos_key = file_data[0]["cosKey"]
                zlFileId = file_data[0]["zlFileId"]
                upload_to_cos(result_file_path, cos_key)

                detection_result_url = f"{self.host}/cycm-cloud/api/detectionResult/insert"
                detection_result = {
                    "label": "stove_unattended",
                    "dataId": frame_data["shop_id"],
                    "detectionType": "stove_unattended",
                    "eventType": "stove_unattended",
                    "score": float(box.conf[0]),
                    "cameraId": camera_id,
                    "fileId": zlFileId,
                }
                response = requests.post(detection_result_url, json=detection_result, headers=headers)

                if response.json().get("result"):
                    resource_data = {"relId": response.json()["result"], "fileList": [zlFileId]}
                    requests.post(
                        f"{self.host}/cloud-service/api/file/upload/fileRelevanceResource",
                        json=resource_data,
                        headers=headers,
                    )

                print(f"[{_ts()}] [STOVE DONE] 摄像头 [{camera_name}](shop={shop_id}) 人离火违规已上报: {label_text}")
            except Exception as e:
                print(
                    f"[{_ts()}] [STOVE ERROR] 摄像头 [{camera_name}](shop={shop_id}) 处理人离火结果异常: {type(e).__name__} - {e}"
                )

    # 方法二：对截图数据进行识别和结果处理
    def process_detection_on_captured_frames(self, rule):
        """
        对存储在对象中的截图数据进行识别处理，并上传结果
        合并了原来的 process_detection_results 功能
        """
        # 获取当前需要处理的截图数据
        frames_to_process = self.captured_frames.copy()

        if not frames_to_process:
            # 仅在处理第一个规则时打印，避免重复信息
            if rule["id"] == 1:
                print("没有需要处理的截图数据")
                return

            # 仅在处理第一个规则时打印，避免重复信息
        if rule["id"] == 1:
            print(f"开始处理 {len(frames_to_process)} 个截图的基本检测和二级检测")

        for camera_id, frame_data in frames_to_process.items():
            try:
                file_path = frame_data["file_path"]
                shop_id = frame_data["shop_id"]
                camera_name = frame_data.get("camera_name", camera_id)
                detection_models = frame_data["detection_models"]

                model = rule["model"]
                classes = rule["classes"]
                conf = rule["conf"]
                print(f"[{_ts()}] [DETECT] 规则 '{rule['label']}' - 摄像头 [{camera_name}](shop={shop_id})")
                if not str(detection_models).__contains__(model):
                    print(f"[{_ts()}] [SKIP] 摄像头 [{camera_name}](shop={shop_id}) 未配置模型 '{model}'，跳过")
                    continue

                # --- 对截图执行需要二级检测的类别：垃圾桶未盖盖 ---
                if "not_close_lid" in model:
                    self.process_two_stage_detection(frame_data, camera_id, "garbage_bin_lid_group", conf)
                    continue
                # --- 对截图执行需要二级检测的类别：人员未穿工作服 ---
                if "without_work_clothes" in model:
                    self.process_two_stage_detection(frame_data, camera_id, "person_work_clothes_group", conf)
                    continue
                # --- 赤膊检测 ---
                if "bareness" in model:
                    self.process_bareness_detection(frame_data, camera_id, conf)
                    continue
                # --- 人离火检测 ---
                if "stove_unattended" in model:
                    self.process_stove_detection(frame_data, camera_id, conf)
                    continue
                # 获取模型（使用缓存的模型）
                if model == "smoke":
                    yolo_mode = self.model_cache["smoke"]
                elif model == "phone":
                    yolo_mode = self.model_cache["phone"]
                else:
                    yolo_mode = self.model_cache["kitchen"]

                # 进行识别
                results = yolo_mode(file_path, stream=True, classes=classes, conf=conf)

                # 收集检测结果，按label去重（保留最高置信度的结果）
                detected_labels = {}

                for result in results:
                    if result.boxes is not None:
                        class_indices = result.boxes.cls  # 类别索引
                        confs = result.boxes.conf  # 置信度
                        names = result.names  # 类别名称字典

                        for cls_idx, conf in zip(class_indices, confs):
                            detected_label = names[int(cls_idx)]
                            # 转为 float
                            conf_value = float(conf)
                            print(
                                f"[{_ts()}] [DETECT] 摄像头 {camera_id} 检测到: {detected_label} conf={conf_value:.2f}"
                            )

                            # 如果这个label还没有记录，先保存图片
                            if detected_label not in detected_labels:
                                # 保存检测结果图片
                                time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                                file_name = time_str + str(random.randint(0, 999)) + ".png"
                                # 在/result/下创建年月日文件夹
                                result_file_path = "./result/" + time.strftime("%Y%m%d", time.localtime()) + "/"
                                os.makedirs(result_file_path, exist_ok=True)
                                # 为每个label保存一张单独的图片
                                save_path = os.path.join(result_file_path, file_name)
                                result.save(save_path)
                                detected_labels[detected_label] = {
                                    "conf": conf_value,
                                    "cls_idx": int(cls_idx),
                                    "file_path": save_path,
                                    "file_name": file_name,
                                }
                            # 更新置信度（保留最高的）
                            if conf_value > detected_labels[detected_label]["conf"]:
                                detected_labels[detected_label]["conf"] = conf_value

                # 处理检测结果：上传图片并创建检测记录（合并后的逻辑）
                for detected_label, detection_info in detected_labels.items():
                    try:
                        conf_value = detection_info["conf"]
                        result_file_path = detection_info["file_path"]
                        result_file_name = detection_info["file_name"]

                        # 上传文件到云存储
                        file_url = f"{self.host}/cloud-service/api/file/upload/buildKeysVo"
                        file_obj = {
                            "files": [
                                {
                                    "name": result_file_name,
                                    "busiScene": "19_31",
                                    "fileName": result_file_name,
                                    "permission": 2,
                                }
                            ]
                        }
                        response = requests.post(file_url, json=file_obj, headers=headers)
                        file_data = response.json()["result"]["files"]
                        cos_key = file_data[0]["cosKey"]
                        zlFileId = file_data[0]["zlFileId"]
                        upload_to_cos(result_file_path, cos_key)
                        print(
                            print(
                                f"[{_ts()}] [UPLOAD OK] 摄像头 [{camera_name}](shop={shop_id}) 图片上传成功: {result_file_name} label={detected_label} conf={conf_value:.2f}"
                            )
                        )

                        # 创建检测记录
                        detection_result_url = f"{self.host}/cycm-cloud/api/detectionResult/insert"
                        detection_result = {
                            "label": detected_label,
                            "dataId": shop_id,
                            "detectionType": detected_label,
                            "eventType": detected_label,
                            "score": conf_value,
                            "cameraId": camera_id,
                        }
                        response = requests.post(detection_result_url, json=detection_result, headers=headers)

                        # 文件关联（每个检测记录关联对应的图片）
                        if response.json().get("result"):
                            resource_data = {"relId": response.json()["result"], "fileList": [zlFileId]}
                            file_relevance_resource_url = (
                                f"{self.host}/cloud-service/api/file/upload/fileRelevanceResource"
                            )
                            requests.post(file_relevance_resource_url, json=resource_data, headers=headers)

                        print(
                            f"[{_ts()}] [DETECT DONE] 摄像头 [{camera_name}](shop={shop_id}) 检测记录已保存: {detected_label} conf={conf_value:.2f}"
                        )

                    except Exception as e:
                        print(
                            f"[{_ts()}] [DETECT ERROR] 摄像头 [{camera_name}](shop={shop_id}) 处理检测结果 '{detected_label}' 异常: {type(e).__name__} - {e}"
                        )

            except Exception as e:
                print(f"[{_ts()}] [ERROR] 摄像头 [{camera_name}](shop={shop_id}) 处理异常: {type(e).__name__} - {e}")

        print(f"[{_ts()}] [DETECT END] 规则 '{rule['label']}' 所有摄像头检测完成")

    def test_one(self):
        """测试函数"""
        yolo_mode = YOLO("./yolo11_kitchen.pt")
        results = yolo_mode("./img/17548959984.jpg", stream=True)
        for result in results:
            if result.boxes is not None:
                class_indices = result.boxes.cls
                confs = result.boxes.conf
                names = result.names
                for cls_idx, conf in zip(class_indices, confs):
                    label = names[int(cls_idx)]
                    print(f"{int(cls_idx)}-{label}: {conf:.2f}")

    def _query_seconds_to_next_rule(self) -> int:
        """查询距离下一条规则到期还有多少秒，供主循环动态调整 sleep，避免频繁无效轮次"""
        try:
            conn = self.pool.connection()
            cs = conn.cursor()
            cs.execute(
                "SELECT TIMESTAMPDIFF(SECOND, NOW(), MIN(c.last_execution_time)) "
                "FROM pgy_server.work_iot_warn_rule r "
                "INNER JOIN pgy_server.work_ai_model_config c ON c.id = r.ai_mode_config_id "
                "WHERE r.enabled = 1 AND c.last_execution_time IS NOT NULL AND c.last_execution_time > NOW()"
            )
            row = cs.fetchone()
            cs.close()
            conn.close()
            if row and row[0] is not None:
                return max(0, int(row[0]))
        except Exception:
            pass
        return 0

    def update_last_update_time(self, rule):
        conn = self.pool.connection()
        cs = conn.cursor()
        # 优先用 DB 查出的 config_id（work_ai_model_config.id），回退到 rule["id"]
        config_id = rule.get("config_id", rule["id"])
        sql = "UPDATE pgy_server.work_ai_model_config SET last_execution_time = DATE_ADD(NOW(), INTERVAL cycle SECOND) WHERE id = %s"
        cs.execute(sql, (config_id,))
        conn.commit()
        cs.close()
        conn.close()

    def run(self) -> int:
        """主运行函数 - 循环逐个处理规则
        返回值：建议的下次等待秒数（0 表示交由主循环的时段间隔决定）"""
        t_run_start = time.time()
        print(f"\n[{_ts()}] [RUN START] {'=' * 50}")
        print(f"[{_ts()}] [RUN START] 新一轮检测任务开始")
        rule_list = [
            {"id": 1, "label": "未戴口罩", "model": "without_mask", "conf": 0.5, "classes": [6]},
            {"id": 2, "label": "未戴厨师帽", "model": "no_hat", "conf": 0.5, "classes": [3]},
            {"id": 8, "label": "乱堆乱放", "model": "garbage", "conf": 0.1, "classes": [10]},
            {"id": 9, "label": "玩手机", "model": "phone", "conf": 0.2, "classes": [0]},
            {"id": 10, "label": "有害生物", "model": "rat", "conf": 0.8, "classes": [4]},
            {"id": 3, "label": "吸烟", "model": "smoke", "conf": 0.5, "classes": [0]},
            {"id": 16, "label": "未穿工作服", "model": "without_work_clothes", "conf": 0.4, "classes": [0]},
            {"id": 15, "label": "垃圾桶未盖盖", "model": "not_close_lid", "conf": 0.4, "classes": [0]},
            {"id": 17, "label": "赤膊", "model": "bareness", "conf": 0.6, "classes": [0]},
            {"id": 18, "label": "人离火", "model": "stove_unattended", "conf": 0.5, "classes": [0]},
        ]
        rule_ids = []
        # 查询对应的模型是否配置到了摄像头上面
        for rule in rule_list:
            sql = f"SELECT * FROM pgy_shop_camera WHERE detection_models LIKE '%{rule['model']}%'"
            conn = self.pool.connection()
            cs = conn.cursor()
            cs.execute(sql)
            data = cs.fetchall()
            if len(data) > 0:
                rule_ids.append(rule["id"])
        print(f"[{_ts()}] [DB] 摄像头已配置的规则 ID: {rule_ids} (共 {len(rule_ids)} 条)")
        # 查询需要执行的规则（r.id=规则ID, c.threshold_min=置信度, c.id=模型配置ID）
        sql = "SELECT r.id, c.threshold_min, c.id FROM work_iot_warn_rule r INNER JOIN work_ai_model_config c on c.id=r.ai_mode_config_id WHERE r.enabled = 1 AND (c.last_execution_time IS NULL OR NOW() >= DATE_ADD(c.last_execution_time, INTERVAL c.cycle SECOND))"
        if len(rule_ids) > 0:
            sql += f" AND r.id IN ({','.join([str(id) for id in rule_ids])})"
        conn = self.pool.connection()
        cs = conn.cursor()
        cs.execute(sql)
        data = cs.fetchall()
        if not data or len(data) == 0:
            secs = self._query_seconds_to_next_rule()
            print(f"[{_ts()}] [SKIP] 当前无待执行规则，下次检查约 {secs}s 后")
            return secs
        print(f"[{_ts()}] [DB] 查询到 {len(data)} 条待执行规则: {[d[0] for d in data]}")
        # 直接调用重构后的方法，不使用线程
        self.capture_frames_from_cameras()
        # 截完图之后再次查询一遍，确保数据是最新的

        cs = conn.cursor()
        cs.execute(sql)
        data = cs.fetchall()
        if not data or len(data) == 0:
            secs = self._query_seconds_to_next_rule()
            print(f"[{_ts()}] [SKIP] 截图后规则已全部过期，下次检查约 {secs}s 后")
            cs.close()
            conn.close()
            return secs
        # d[0]=r.id, d[1]=c.threshold_min(置信度), d[2]=c.id(模型配置ID)
        rule_id_map = {d[0]: {"conf": d[1], "config_id": d[2]} for d in data}

        # 循环逐个处理规则
        processed_count = 0
        for rule in rule_list:
            if rule["id"] not in rule_id_map:
                print(f"[{_ts()}] [SKIP] 规则 '{rule['label']}' (ID={rule['id']}) 未到执行周期，跳过")
                continue
            t_rule_start = time.time()
            entry = rule_id_map[rule["id"]]
            # DB 有值则用 DB 的置信度，否则回退到硬编码默认值
            conf_value = float(entry["conf"]) if entry["conf"] is not None else rule["conf"]
            rule["conf"] = conf_value
            rule["config_id"] = entry["config_id"]
            print(
                f"[{_ts()}] [RULE START] 开始处理规则: '{rule['label']}' (ID={rule['id']}, model={rule['model']}, conf={conf_value})"
            )
            self.process_detection_on_captured_frames(rule)
            elapsed = time.time() - t_rule_start
            print(f"[{_ts()}] [RULE DONE] 规则 '{rule['label']}' 处理完成，耗时 {elapsed:.2f}s")
            processed_count += 1
            self.update_last_update_time(rule)
        self.captured_frames.clear()  # 清空已处理的数据
        cs.close()
        conn.close()
        total_elapsed = time.time() - t_run_start
        print(f"[{_ts()}] [RUN END] 本轮任务结束，共处理 {processed_count} 条规则，总耗时 {total_elapsed:.2f}s")
        print(f"[{_ts()}] [RUN END] {'=' * 50}\n")


def main(environment=None):
    """主函数 - 创建检测系统实例并运行"""
    detection_system = DetectionSystem(environment)
    detection_system.run()


if __name__ == "__main__":
    # 1. 在循环外初始化：确保 10 几个模型和数据库连接池只加载 1 次！
    detection_system = DetectionSystem("test")

    # 2. 只有检测逻辑在无限循环中运行
    while True:
        t_start = time.time()
        suggested_sleep = 0
        try:
            suggested_sleep = detection_system.run() or 0
        except Exception as e:
            # 加上全局异常捕获，防止因为偶尔的网络抖动导致整个脚本崩溃退出
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [CRITICAL] 主循环发生致命异常: {type(e).__name__} - {e}"
            )
        interval = get_detection_interval()
        elapsed = time.time() - t_start
        # 取"时段间隔"和"DB建议等待"中的较大值，避免规则未到期时的无效轮次
        effective_interval = max(interval, suggested_sleep)
        sleep_secs = max(0.0, effective_interval - elapsed)
        next_run = time.strftime("%H:%M:%S", time.localtime(time.time() + sleep_secs))
        print(
            f"[{_ts()}] [SCHEDULER] 间隔={effective_interval // 60}分钟 | 本轮耗时={elapsed:.1f}s | "
            f"等待={sleep_secs:.1f}s | 下次执行时间={next_run}"
        )
        time.sleep(sleep_secs)
