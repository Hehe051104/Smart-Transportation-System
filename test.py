import sqlite3
import os

DB_NAME = "drone_data.db"

def view_data():
    if not os.path.exists(DB_NAME):
        print(f"❌ 错误: 找不到数据库文件 {DB_NAME}")
        print("请先运行 main.py 让无人机生成数据，或者运行 create_fake_data.py 生成测试数据。")
        return

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # 1. 查询总数量
        c.execute("SELECT COUNT(*) FROM flight_logs")
        count = c.fetchone()[0]
        print(f"📊 当前数据库中共有 {count} 条记录")
        print("-" * 50)

        if count == 0:
            print("⚠️ 数据库是空的。")
        else:
            # 2. 查询最新的 10 条数据 (按 id 倒序)
            print("📋 最新写入的 10 条数据:")
            print(f"{'ID':<5} | {'时间 (Time)':<20} | {'纬度 (Lat)':<10} | {'经度 (Lon)':<10} | {'高度 (Alt)'}")
            print("-" * 60)

            c.execute("SELECT * FROM flight_logs ORDER BY id DESC LIMIT 10")
            rows = c.fetchall()

            for row in rows:
                # row[0]=id, row[1]=timestamp, row[2]=lat, row[3]=lon, row[4]=alt
                print(f"{row[0]:<5} | {row[1]:<20} | {row[2]:<10.6f} | {row[3]:<10.6f} | {row[4]}")

        conn.close()

    except Exception as e:
        print(f"❌ 读取数据库出错: {e}")

if __name__ == "__main__":
    view_data()