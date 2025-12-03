import sqlite3
import datetime
import math

# 数据库文件名 (必须和 main.py 里的一致)
DB_NAME = "drone_data.db"

def create_fake_data():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. 确保表存在 (防止你还没运行过 main.py)
    c.execute('''CREATE TABLE IF NOT EXISTS flight_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  lat REAL,
                  lon REAL,
                  alt REAL)''')

    # 2. 模拟参数
    # 苏州中心坐标 (WGS84)
    center_lat = 31.299379
    center_lon = 120.619585
    radius = 0.004  # 半径约 400米

    # 设定起始时间为：当前时间往前推 1 小时
    start_time = datetime.datetime.now() - datetime.timedelta(hours=1)

    print(f"💾 正在连接数据库: {DB_NAME}")
    print("🛠️ 正在生成 200 个模拟轨迹点 (圆形路径)...")

    points_count = 200
    for i in range(points_count):
        # 计算圆周运动坐标
        angle = (2 * math.pi / points_count) * i

        # 简单的经纬度偏移算法
        # 注意：这里的 lat/lon 是模拟的 WGS84 坐标
        lat = center_lat + radius * math.sin(angle)
        lon = center_lon + radius * math.cos(angle)
        alt = 50.0  # 假设高度 50米

        # 时间递增 (每隔 5 秒一个点)
        point_time = start_time + datetime.timedelta(seconds=i*5)

        # 插入数据库
        c.execute("INSERT INTO flight_logs (timestamp, lat, lon, alt) VALUES (?, ?, ?, ?)",
                  (point_time.isoformat(), lat, lon, alt))

    conn.commit()
    conn.close()

    # 计算结束时间用于提示
    end_time = start_time + datetime.timedelta(seconds=points_count*5)

    print("-" * 40)
    print("✅ 数据生成成功！")
    print("请在前端页面 [历史轨迹回放] 区域选择以下时间范围：")
    print(f"👉 开始时间: {start_time.strftime('%Y-%m-%dT%H:%M')}")
    print(f"👉 结束时间: {end_time.strftime('%Y-%m-%dT%H:%M')}")
    print("-" * 40)

if __name__ == "__main__":
    create_fake_data()