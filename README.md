# 智慧交通系统（YOLO 目标检测版）

本项目使用 FastAPI 提供视频流与无人机状态服务，并在视频帧中用 YOLO 检测行人和汽车。

## 功能
- 视频流实时输出（`/video_feed`）
- YOLO 目标检测（行人、车辆）
- 历史轨迹查询（`/api/history`）

## 运行前准备
- Python 3.8+
- 建议先创建虚拟环境

## 安装依赖
```powershell
pip install -r requirements.txt
```

## 启动服务
```powershell
python .\main.py
```

## 说明
- 默认使用同级目录 `test.mp4` 作为视频源（可在 `main.py` 修改 `USE_TEST_VIDEO`）。
- YOLO 模型默认使用 `yolov8n.pt`，首次运行会自动下载模型。

## 常见问题
- 若无法下载模型，请手动将 `yolov8n.pt` 放入 `models/yolo/` 目录。
