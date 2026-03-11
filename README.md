# 智慧交通系统

基于 `FastAPI + OpenCV + YOLO + DroneKit + 百度地图` 的无人机智慧交通演示项目。

系统可以接入无人机遥测数据与视频流，对视频中的 **行人**、**车辆** 进行实时识别，并将 **当前无人机坐标 + 检测统计** 保存到本地数据库；随后可在历史回放中查看轨迹，并在鼠标悬停到历史点时显示该时刻的坐标、时间与检测结果。

## 主要功能

- 无人机实时坐标接入与状态展示
- 无人机 RTSP 视频流接入
- YOLO 实时目标检测
- 行人 / 车辆计数
- 飞行轨迹持久化存储
- 检测结果与当前无人机坐标绑定入库
- 历史轨迹查询与地图回放
- 鼠标悬停历史点显示：坐标、时间、行人/车辆数量
- 右侧实时流量图表联动显示

## 项目结构

当前仓库中和运行相关的核心文件如下：

- `main.py`：后端主程序，包含无人机连接、视频处理、YOLO 推理、数据库写入、API 服务
- `index.html`：前端页面，包含地图、视频监控、历史轨迹回放、实时流量图表
- `requirements.txt`：Python 依赖清单
- `requirement.txt`：额外生成的一份依赖清单（与 `requirements.txt` 内容一致）
- `drone_data.db`：SQLite 数据库文件
- `test.mp4`：本地测试视频
- `models/`：模型目录

## 技术栈

- Python
- FastAPI
- Uvicorn
- OpenCV
- Ultralytics YOLO
- DroneKit
- SQLite
- 百度地图 JavaScript API
- Chart.js
- Tailwind CSS

## 运行前准备

### 环境要求

- Python 3.8 及以上
- Windows / Linux 均可（当前项目主要按 Windows 环境使用）
- 能访问无人机遥测链路
- 能访问无人机 RTSP 视频流

### 建议

- 推荐使用虚拟环境
- 首次运行前确认 `opencv-python`、`dronekit`、`ultralytics` 已正确安装
- 若使用 GPU 推理，请确保 CUDA 环境可用

## 安装依赖

```powershell
pip install -r requirements.txt
```

如果你希望使用你新生成的文件，也可以：

```powershell
pip install -r requirement.txt
```

## 启动方式

```powershell
python .\main.py
```

启动后访问：

- 前端页面：`http://localhost:8000`
- 视频流接口：`http://localhost:8000/video_feed`
- 无人机状态接口：`http://localhost:8000/api/drone_status`
- 检测计数接口：`http://localhost:8000/api/detection_counts`
- 历史轨迹接口：`http://localhost:8000/api/history?start_time=2026-03-11T10:00&end_time=2026-03-11T11:00`

## 当前默认配置

以下配置位于 `main.py` 顶部：

- `RTSP_URL = "rtsp://192.168.1.202:8554/video"`
- `USE_TEST_VIDEO = False`
- `DRONE_CONNECTION_STRING = 'udp:192.168.1.123:14550'`
- `DB_NAME = "drone_data.db"`
- `YOLO_MODEL_NAME = "yolov8s.pt"`

### 配置说明

#### 1. 视频源

当前默认使用无人机 RTSP 实时流：

- 当 `USE_TEST_VIDEO = False` 时，视频源使用 `RTSP_URL`
- 当 `USE_TEST_VIDEO = True` 时，视频源切换为本地 `test.mp4`

也就是说：

- 联机使用无人机时，保持 `False`
- 本地调试或演示时，可改成 `True`

#### 2. 无人机连接

当前通过：

`udp:192.168.1.123:14550`

接入无人机遥测数据。若你的设备地址不同，请在 `main.py` 中修改 `DRONE_CONNECTION_STRING`。

#### 3. 模型配置

当前检测类别为：

- `person`
- `car`

对应代码中的：

```python
YOLO_CLASSES = [0, 2]
```

其中：

- `0` = person
- `2` = car

## 数据库说明

项目使用 SQLite，本地数据库文件为：

`drone_data.db`

### 表结构

#### `flight_logs`

用于保存无人机飞行轨迹：

- `timestamp`
- `lat`
- `lon`
- `alt`

#### `detection_logs`

用于保存检测结果与当前无人机坐标：

- `timestamp`
- `lat`
- `lon`
- `person_count`
- `car_count`
- `total_count`

## 功能链路说明

### 1. 无人机坐标记录

后端线程会持续从无人机读取经纬度，并写入 `flight_logs`。

### 2. 视频检测

系统读取 RTSP 视频流，对画面中的目标做 YOLO 推理，统计：

- 行人数量
- 车辆数量

### 3. 检测结果入库

每次检测完成后，程序会把：

- 当前无人机坐标
- 当前统计到的行人数量
- 当前统计到的车辆数量

一起写入 `detection_logs`。

### 4. 历史回放

前端查询指定时间范围后：

- 后端读取 `flight_logs`
- 再读取 `detection_logs`
- 按时间进行合并
- 返回给前端进行地图绘制

在地图中鼠标悬停到历史点时，会显示：

- 坐标
- 时间
- 行人数量
- 车辆数量

## 前端页面说明

系统页面包含以下区域：

### 左侧：道路精准标红

- 可输入道路名
- 支持通过地图接口获取路网形状

### 右侧：无人机实时状态

显示：

- 当前经纬度
- 连接状态

### 右侧：历史轨迹回放

支持：

- 选择开始时间
- 选择结束时间
- 查询轨迹
- 清除轨迹

### 右侧：实时流量

- 非历史模式下：显示当前实时检测到的行人/车辆数量
- 历史模式下：当鼠标悬停某个历史点时，显示该点对应的行人/车辆数据

### 左下角：实时监控窗口

- 显示无人机视频流
- 绘制 YOLO 检测框
- 支持拖拽
- 支持缩放

## 推荐使用流程

### 联机实飞 / 联机测试

1. 确认无人机遥测地址可达
2. 确认 RTSP 视频源可达
3. 启动 `main.py`
4. 打开浏览器进入 `http://localhost:8000`
5. 观察实时画面、地图、右侧状态面板
6. 飞行结束后，在历史回放中选择时间范围进行查询
7. 鼠标悬停历史点查看坐标、时间、车辆/行人数据

### 本地调试

1. 在 `main.py` 中将 `USE_TEST_VIDEO = True`
2. 准备好 `test.mp4`
3. 启动服务并观察检测效果

## 注意事项

### 1. 当前检测逻辑与视频流访问有关

当前实现下，检测处理是在 `/video_feed` 被访问时运行的。

这意味着：

- 打开前端页面并加载监控画面时，检测会运行
- 若完全不访问视频流页面，检测逻辑不会持续处理视频帧

对于当前单人演示/调试场景，这是可用的。

### 2. 不建议同时打开多个页面

如果多个页面同时请求 `/video_feed`，可能会重复启动视频处理流程，增加计算负担。

### 3. 历史数据匹配是按时间合并

回放时，后端会将飞行轨迹与检测记录按时间进行合并，因此展示效果满足当前项目需要；但它并不是基于“逐帧精确绑定”的工业级时序方案。

## 常见问题

### 1. `No module named 'cv2'`

说明未安装 OpenCV：

```powershell
pip install opencv-python
```

### 2. YOLO 模型加载失败

可检查：

- 是否已安装 `ultralytics`
- 模型文件是否存在
- 网络是否允许首次下载模型

当前项目代码默认使用：

`yolov8s.pt`

### 3. 无法连接无人机

检查：

- `DRONE_CONNECTION_STRING` 是否正确
- 遥测链路是否连通
- 端口是否被占用

### 4. 无法打开 RTSP 视频流

检查：

- `RTSP_URL` 是否正确
- 无人机是否正在推流
- 本机网络是否能访问图传地址
- OpenCV / FFMPEG 是否支持该流格式

### 5. 历史轨迹查不到点

检查：

- 当时是否真的有飞行坐标写入 `flight_logs`
- 经纬度是否为有效值（非 0,0）
- 时间范围是否选择正确

### 6. 历史点上没有车辆/行人数据

检查：

- 当时页面是否打开并在拉取 `/video_feed`
- 当时是否检测到目标
- `detection_logs` 是否写入成功

## 后续可优化方向

- 将视频检测改为全局后台线程，避免依赖页面是否打开
- 将 `RTSP_URL`、`DRONE_CONNECTION_STRING` 抽到配置文件
- 为数据库查询增加更稳定的索引与聚合逻辑
- 增加接口健康检查
- 增加自动化测试与数据校验脚本

## 许可证

当前仓库未声明许可证；如需开源发布，建议补充 `LICENSE` 文件。
