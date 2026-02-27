import collections
import collections.abc
# 强行给 collections 模块把 MutableMapping 补回去 (解决 Python 3.10+ dronekit 兼容问题)
collections.MutableMapping = collections.abc.MutableMapping

import cv2
import uvicorn
import threading
import time
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dronekit import connect, VehicleMode
import os
import sys
import queue

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None

app = FastAPI()

# ----------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------
RTSP_URL = "rtsp://192.168.1.202:8554/video"  # 视频流地址
USE_TEST_VIDEO = True  # 暂时使用同级目录下的 test.mp4
VIDEO_SOURCE = "test.mp4" if USE_TEST_VIDEO else RTSP_URL
DRONE_CONNECTION_STRING = 'udp:192.168.1.123:14550'  # 无人机连接地址
DRONE_BAUD = 921600
DB_NAME = "drone_data.db"  # 数据库文件名

DETECTION_ENABLED = True  # 是否在视频流上运行目标检测
MODEL_DIR = "models/yolo"
YOLO_MODEL_NAME = "yolov8s.pt"
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, YOLO_MODEL_NAME)
YOLO_CONFIDENCE = 0.2
DETECTION_STRIDE = 3  # 每隔 N 帧做一次推理，提升识别率
INFER_IMGSZ = 640
JPEG_QUALITY = 55
TARGET_FPS = 12
USE_GPU = True
OUTPUT_MAX_WIDTH = 960
YOLO_CLASSES = [0, 2]  # COCO: person=0, car=2
CAPTURE_QUEUE_SIZE = 2
INFER_QUEUE_SIZE = 1

# 全局变量：实时数据
current_drone_data = {
    "lat": 0.0,
    "lon": 0.0,
    "alt": 0.0,
    "status": "Disconnected"
}

# 检测统计（线程安全）
current_detection_counts = {
    "person": 0,
    "car": 0,
    "total": 0,
    "timestamp": ""
}
counts_lock = threading.Lock()

# ----------------------------------------------------------------
# 数据库操作函数
# ----------------------------------------------------------------
def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS flight_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  lat REAL,
                  lon REAL,
                  alt REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS detection_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  lat REAL,
                  lon REAL,
                  person_count INTEGER,
                  car_count INTEGER,
                  total_count INTEGER)''')
    # 回填旧数据：将 flight_logs 的坐标写入 detection_logs，计数填 0
    c.execute('''
        INSERT INTO detection_logs (timestamp, lat, lon, person_count, car_count, total_count)
        SELECT f.timestamp, f.lat, f.lon, 0, 0, 0
        FROM flight_logs f
        LEFT JOIN detection_logs d
          ON d.timestamp = f.timestamp AND d.lat = f.lat
        WHERE d.id IS NULL
    ''')
    conn.commit()
    conn.close()
    print("💾 [DB] 数据库已就绪")


def save_log(lat, lon, alt):
    """插入一条日志"""
    if abs(lat) < 0.1 and abs(lon) < 0.1:
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now_str = datetime.now().isoformat()
        now_str = now_str.split('.')[0]
        c.execute("INSERT INTO flight_logs (timestamp, lat, lon, alt) VALUES (?, ?, ?, ?)",
                  (now_str, lat, lon, alt))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ [DB] 写入失败: {e}")


def save_detection_log(lat, lon, person_count, car_count, total_count):
    if abs(lat) < 0.1 and abs(lon) < 0.1:
        return
    if total_count <= 0:
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now_str = datetime.now().isoformat(timespec="seconds")
        c.execute(
            "INSERT INTO detection_logs (timestamp, lat, lon, person_count, car_count, total_count) VALUES (?, ?, ?, ?, ?, ?)",
            (now_str, lat, lon, person_count, car_count, total_count)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ [DB] 检测日志写入失败: {e}")


def ensure_yolo_model_dir():
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)


def load_detection_model():
    """加载 YOLO 模型（Ultralytics）。返回 model 或 None。"""
    if not DETECTION_ENABLED:
        return None
    if YOLO is None:
        print("❌ [Model] 未安装 ultralytics，无法加载 YOLO")
        return None
    try:
        ensure_yolo_model_dir()
        model_source = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else YOLO_MODEL_NAME
        model = YOLO(model_source)
        if USE_GPU and torch is not None and torch.cuda.is_available():
            model.to("cuda")
            print("🚀 [Model] 已启用 GPU(CUDA) 推理")
        else:
            print("ℹ️ [Model] 使用 CPU 推理")
        return model
    except Exception as e:
        print(f"❌ [Model] 加载失败: {e}")
        return None


# ----------------------------------------------------------------
# 1. 无人机后台线程 (核心逻辑)
# ----------------------------------------------------------------
def drone_telemetry_loop():
    print(f"🚀 [Drone] 正在尝试连接无人机: {DRONE_CONNECTION_STRING} ...")

    vehicle = None
    try:
        vehicle = connect(DRONE_CONNECTION_STRING, wait_ready=False, baud=DRONE_BAUD)
        print("✅ [Drone] 无人机连接成功！")
        current_drone_data["status"] = "Connected"
    except Exception as e:
        print(f"❌ [Drone] 连接失败: {e}")
        return

    while True:
        try:
            location = vehicle.location.global_frame

            if location.lat is not None and location.lon is not None:
                current_drone_data["lat"] = location.lat
                current_drone_data["lon"] = location.lon

                print(f"📡 [GPS] Lat: {location.lat:.7f}, Lon: {location.lon:.7f}")

                save_log(location.lat, location.lon, location.alt or 0)

            time.sleep(1.0)

        except Exception:
            time.sleep(1)


# ----------------------------------------------------------------
# 2. Web 服务器逻辑
# ----------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ [Video] 无法连接视频流: {VIDEO_SOURCE}")
        return

    model = None
    if DETECTION_ENABLED:
        model = load_detection_model()
        if model is None:
            print("⚠️ [Model] 未能加载检测模型，继续不进行检测")
        else:
            try:
                if USE_GPU and torch is not None and torch.cuda.is_available():
                    model.fuse()
            except Exception:
                pass

    frame_queue = queue.Queue(maxsize=CAPTURE_QUEUE_SIZE)
    infer_queue = queue.Queue(maxsize=INFER_QUEUE_SIZE)
    last_boxes = []
    last_boxes_lock = threading.Lock()

    def capture_loop():
        while True:
            success, frame = cap.read()
            if not success:
                if USE_TEST_VIDEO:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if OUTPUT_MAX_WIDTH and frame.shape[1] > OUTPUT_MAX_WIDTH:
                scale = OUTPUT_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (OUTPUT_MAX_WIDTH, int(frame.shape[0] * scale)))

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

            if DETECTION_ENABLED and model is not None:
                if infer_queue.full():
                    try:
                        infer_queue.get_nowait()
                    except queue.Empty:
                        pass
                infer_queue.put(frame)

    def infer_loop():
        nonlocal last_boxes
        frame_idx = 0
        while True:
            if not DETECTION_ENABLED or model is None:
                time.sleep(0.05)
                continue

            try:
                frame = infer_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            frame_idx += 1
            if frame_idx % DETECTION_STRIDE != 0:
                continue

            try:
                use_cuda = USE_GPU and torch is not None and torch.cuda.is_available()
                results = model.predict(
                    frame,
                    imgsz=INFER_IMGSZ,
                    conf=YOLO_CONFIDENCE,
                    classes=YOLO_CLASSES,
                    verbose=False,
                    device=0 if use_cuda else "cpu",
                    half=True if use_cuda else False,
                )
                boxes = []
                person_count = 0
                car_count = 0
                if results:
                    result = results[0]
                    names = result.names or {}
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        label = names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes.append((label, conf, x1, y1, x2, y2))
                        if label == "person":
                            person_count += 1
                        elif label == "car":
                            car_count += 1

                with last_boxes_lock:
                    last_boxes = boxes

                with counts_lock:
                    current_detection_counts["person"] = person_count
                    current_detection_counts["car"] = car_count
                    current_detection_counts["total"] = person_count + car_count
                    current_detection_counts["timestamp"] = datetime.now().isoformat(timespec="seconds")

                # 写入数据库（绑定当前无人机坐标）
                save_detection_log(
                    current_drone_data.get("lat", 0.0),
                    current_drone_data.get("lon", 0.0),
                    person_count,
                    car_count,
                    person_count + car_count
                )
            except Exception as e:
                print(f"⚠️ [Model] 推理错误: {e}")

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=infer_loop, daemon=True).start()

    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with last_boxes_lock:
            boxes_snapshot = list(last_boxes)

        for label, conf, x1, y1, x2, y2 in boxes_snapshot:
            startX, startY, endX, endY = map(int, [x1, y1, x2, y2])
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(frame.shape[1] - 1, endX), min(frame.shape[0] - 1, endY)
            color = (0, 0, 255) if label == 'person' else (0, 255, 0)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            text = f"{label}: {conf:.2f}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        if TARGET_FPS > 0:
            time.sleep(1.0 / TARGET_FPS)

    cap.release()


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/api/drone_status")
async def get_drone_status():
    return current_drone_data


@app.get("/api/detection_counts")
async def get_detection_counts():
    with counts_lock:
        return dict(current_detection_counts)


@app.get("/api/history")
def get_history(start_time: str, end_time: str):
    """
    获取历史轨迹
    参数格式: 2023-10-27T10:00 (ISO 8601)
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # 1. 获取飞行日志
        query_flight = """
            SELECT timestamp, lat, lon
            FROM flight_logs
            WHERE timestamp BETWEEN ? AND ?
              AND (lat > 0.1 OR lat < -0.1)
            ORDER BY timestamp ASC
        """
        c.execute(query_flight, (start_time, end_time))
        flights = [dict(r) for r in c.fetchall()]

        # 2. 获取检测日志
        query_detection = """
            SELECT timestamp, person_count, car_count
            FROM detection_logs
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """
        c.execute(query_detection, (start_time, end_time))
        detections = [dict(r) for r in c.fetchall()]
        
        conn.close()

        # 3. 数据合并 (以时间戳秒级匹配)
        merged_data = []

        # 构建检测数据索引： timestamp(秒) -> max(counts)
        det_map = {}
        for d in detections:
            # 时间戳可能带毫秒，截断到秒
            t_key = d['timestamp'].split('.')[0]
            current_total = d['person_count'] + d['car_count']
            
            if t_key not in det_map:
                det_map[t_key] = d
            else:
                # 如果同一秒有多条检测记录，保留统计数更多的那条
                existing = det_map[t_key]
                if current_total > (existing['person_count'] + existing['car_count']):
                    det_map[t_key] = d

        # 遍历飞行轨迹进行匹配
        for f in flights:
            t_str = f['timestamp']
            t_key = t_str.split('.')[0]
            
            item = {
                "lat": f["lat"], 
                "lon": f["lon"], 
                "time": t_key.replace('T', ' '),
                "person_count": 0,
                "car_count": 0
            }

            if t_key in det_map:
                d = det_map[t_key]
                item["person_count"] = d['person_count']
                item["car_count"] = d['car_count']
            
            merged_data.append(item)

        return {"status": "success", "data": merged_data}

    except Exception as e:
        print(f"History query error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/")
async def read_index():
    response = FileResponse('index.html')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


app.mount("/", StaticFiles(directory="."), name="static")

# ----------------------------------------------------------------
# 3. 程序入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    init_db()

    t = threading.Thread(target=drone_telemetry_loop, daemon=True)
    t.start()

    print("🌐 [Web] 服务器启动中... http://localhost:8000")
    # 禁用 uvicon 自身的 access log 以减少干扰，reload=False 生产模式
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)



