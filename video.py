import cv2

def play_rtsp_stream():
    # 你的 RTSP 地址
    rtsp_url = "rtsp://192.168.1.202:8554/video"

    print(f"正在尝试连接: {rtsp_url} ...")

    # 创建视频捕获对象
    # cv2.CAP_FFMPEG 显式告诉 OpenCV 使用 FFMPEG 后端（通常能提高兼容性）
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 设置缓冲区大小为1，以减少延迟（可选，视具体情况而定）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 检查是否成功打开
    if not cap.isOpened():
        print("错误: 无法打开视频流。请检查：")
        print("1. 无人机/摄像头是否已开机并连接网络")
        print("2. 电脑是否与设备在同一个局域网 (192.168.1.x)")
        print("3. RTSP 地址是否正确")
        return

    print("连接成功！按 'q' 键退出播放。")

    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()

            # 如果读取失败（可能是流断了）
            if not ret:
                print("无法获取画面 (流可能已中断)")
                break

            # 显示画面
            # 你可以在这里调整窗口大小，例如: cv2.resize(frame, (640, 480))
            cv2.imshow('RTSP Video Stream', frame)

            # 按 'q' 键退出循环
            # waitKey(1) 表示延时1ms，数值越小越流畅
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("用户强制中断")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("播放结束")

if __name__ == "__main__":
    play_rtsp_stream()