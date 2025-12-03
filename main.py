import collections
import collections.abc
# 强行给 collections 模块把 MutableMapping 补回去 (解决 Python 3.10+ 兼容性)
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

app = FastAPI()

# ----------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------
RTSP_URL = "rtsp://192.168.1.202:8554/video"  # 视频流地址
DRONE_CONNECTION_STRING = 'udp:192.168.1.123:14550' # 无人机连接地址
DRONE_BAUD = 921600
DB_NAME = "drone_data.db"  # 数据库文件名

# 全局变量：实时数据
current_drone_data = {
    "lat": 0.0,
    "lon": 0.0,
    "alt": 0.0,
    "status": "Disconnected"
}

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
    conn.commit()
    conn.close()
    print("💾 [DB] 数据库已就绪")

def save_log(lat, lon, alt):
    """插入一条日志"""
    # 简单的过滤：如果是 0.0 就不存了，节省空间
    if abs(lat) < 0.1 and abs(lon) < 0.1:
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now_str = datetime.now().isoformat()
        # 截取到秒，不用太精确
        now_str = now_str.split('.')[0]
        c.execute("INSERT INTO flight_logs (timestamp, lat, lon, alt) VALUES (?, ?, ?, ?)",
                  (now_str, lat, lon, alt))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ [DB] 写入失败: {e}")

# ----------------------------------------------------------------
# 1. 无人机后台线程 (核心逻辑)
# ----------------------------------------------------------------
def drone_telemetry_loop():
    print(f"🚀 [Drone] 正在尝试连接无人机: {DRONE_CONNECTION_STRING} ...")

    vehicle = None
    try:
        # wait_ready=False: 不管有没有报错，只要连上就行
        vehicle = connect(DRONE_CONNECTION_STRING, wait_ready=False, baud=DRONE_BAUD)
        print("✅ [Drone] 无人机连接成功！")
        current_drone_data["status"] = "Connected"
    except Exception as e:
        print(f"❌ [Drone] 连接失败: {e}")
        return

    while True:
        try:
            # 读取位置信息
            location = vehicle.location.global_frame

            if location.lat is not None and location.lon is not None:
                current_drone_data["lat"] = location.lat
                current_drone_data["lon"] = location.lon
                # current_drone_data["alt"] = location.alt

                print(f"📡 [GPS] Lat: {location.lat:.7f}, Lon: {location.lon:.7f}")

                # 保存到数据库 (每1秒存一次)
                save_log(location.lat, location.lon, location.alt or 0)

            time.sleep(1.0)

        except Exception as e:
            # 这里的报错通常是 link timeout，不影响主程序运行
            # print(f"⚠️ [Drone] 读取循环警告: {e}")
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
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ [Video] 无法连接视频流: {RTSP_URL}")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 压缩图片以提高网络传输速度
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")

@app.get("/api/drone_status")
async def get_drone_status():
    return current_drone_data

@app.get("/api/history")
async def get_history(start_time: str, end_time: str):
    """
    获取历史轨迹
    参数格式: 2023-10-27T10:00 (ISO 8601)
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # --- 核心修复：后端过滤掉 lat/lon 为 0 的无效数据 ---
        query = """
                SELECT lat, lon, timestamp
                FROM flight_logs
                WHERE timestamp BETWEEN ? AND ?
                  AND (lat > 0.1 OR lat < -0.1)
                  AND (lon > 0.1 OR lon < -0.1)
                ORDER BY timestamp ASC \
                """

        c.execute(query, (start_time, end_time))
        rows = c.fetchall()
        conn.close()

        data = [{"lat": r[0], "lon": r[1], "time": r[2]} for r in rows]
        return {"status": "success", "count": len(data), "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def read_index():
    return FileResponse('final.html')

app.mount("/", StaticFiles(directory="."), name="static")

# ----------------------------------------------------------------
# 3. 程序入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    # 1. 初始化数据库
    init_db()

    # 2. 启动无人机线程
    t = threading.Thread(target=drone_telemetry_loop, daemon=True)
    t.start()

    # 3. 启动 Web 服务器
    print("🌐 [Web] 服务器启动中... http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)