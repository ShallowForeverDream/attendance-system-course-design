# 内容安全实验课 · 班级智能考勤系统

本项目依据 `6.考勤系统.pptx` 与 `2026《内容安全实验课》课程设计要求及评分标准.docx` 搭建，采用 BS（Browser/Server）架构，实现基础考勤、动作活体检测、人脸库比对、合照学生识别、情绪分析、权限控制、异常处理、Excel 导出与统计报表。

## 1. 技术栈

- 前端：HTML + CSS + JavaScript，浏览器 `getUserMedia` 调用摄像头
- 后端：Flask API
- 数据库：SQLite
- 视觉算法：OpenCV Haar 人脸检测 + 本地 LBP/纹理特征向量匹配 + 动作活体检测 + 轻量情绪启发式分类
- 导出：openpyxl 生成 Excel

> 当前实现优先保证课程验收端到端可运行；若后续允许安装 DeepFace/FER/InsightFace，可在 `core/vision.py` 中替换 `face_embedding()` 与 `analyze_emotion()`，不影响 API 与前端。

## 2. 快速运行

```powershell
cd C:\大学\大三\大三下\内容安全实践\实验六-课程设计\attendance_system
python -m pip install -r requirements.txt
python app.py
```

浏览器打开：<http://127.0.0.1:5000>

默认账号：

| 角色 | 用户名 | 密码 |
|---|---|---|
| 教师 | `teacher` | `teacher123` |
| 学生 | `student01` | `student123` |
| 学生 | `student02` | `student123` |

## 3. 推荐演示流程

1. 教师登录。
2. 先导入老师提供的班级照片：

   ```powershell
   python scripts\import_face_data.py --source ..\face_data
   python scripts\evaluate_face_data.py
   python scripts\make_demo_collage.py
   ```

3. 在“学生/人脸库”中确认学生和人脸样本数，也可新增学生或用摄像头补采样本。
4. 进入“考勤打卡”，开启摄像头，按随机动作挑战完成打卡。
5. 在“考勤记录”中按日期/学号筛选，并导出 Excel。
6. 在“合照识别”上传班级/活动合照，查看标注图、参与名单与情绪结果。
7. 在“安全自测”中运行静态照片攻击自测，证明重复帧会被活体检测拒绝。
8. 在“统计报表”查看活动频次和情绪分布。
9. 使用学生账号登录，验证只能查看本人数据。
10. 打开“验收清单”，逐项展示每个得分点。

## 4. 功能与评分点对应

| 评分点 | 项目实现 |
|---|---|
| BS 分层架构 | `templates/static` 前端、`app.py/core` 后端、SQLite 数据库 |
| 摄像头采集 | `static/js/app.js` 使用 `navigator.mediaDevices.getUserMedia()` |
| 手动/自动捕捉 | “手动抓拍预检”可做单帧质量/识别/情绪预检；“开始活体打卡”按随机动作分段自动采集多帧 |
| 活体检测 | `analyze_liveness()`：随机动作挑战 + 多帧人脸质量/纹理检测 |
| 人脸库 | 学生 CRUD + 查看/删除单个人脸样本 + 多人脸样本上传 + `face_data` 批量导入 + 摄像头补采样本 |
| 考勤记录 | `attendance_records` 表，支持筛选与 Excel 导出 |
| 合照识别 | 上传合照，批量检测人脸、逐个匹配人脸库、生成标注图和名单 |
| 活动频次 | `activity_participants` 汇总统计 |
| 情绪分析 | 考勤/合照过程中记录 `emotion_records` |
| 权限控制 | 教师/学生会话与接口权限装饰器 |
| 异常防护 | 摄像头失败、图片解析失败、识别失败、挑战过期等均返回可读错误 |

## 4.1 现场兜底演示

- 如果现场摄像头被浏览器权限、教室设备或投屏环境影响，可在“安全自测”点击 **无摄像头样本攻击自测**，系统会用已入库样本构造静态照片/重复帧攻击并证明活体检测拒绝。
- 如果老师追问“如何抵御预录视频”，点击 **展示随机挑战抗视频**，可直接展示多组随机动作挑战、90 秒过期和多帧运动检查逻辑。
- “手动抓拍预检”只用于证明前端支持手动拍摄与实时反馈，不写入考勤；正式考勤必须通过随机动作活体检测，避免手动单帧绕过安全逻辑。

## 5. 老师照片数据导入与稳定合照演示

老师提供的 `..\face_data` 命名格式大多为：

```text
学号-姓名-专业-性别.jpg
```

导入脚本会自动解析学号、姓名、专业、性别，创建学生记录，检测人脸，裁剪人脸区域，提取特征并写入 `face_samples`。

生成的报告：

- `docs/face_data_import_report.json`
- `docs/face_data_evaluation_report.json`
- `docs/demo_collage_report.json`

当前默认人脸阈值为 `0.78`，兼顾合照召回率与误识别率。`scripts/make_demo_collage.py` 会生成 10 人 PNG 演示合照，`scripts/group_collage_selftest.py` 会通过系统真实 `/api/group/recognize` 接口自测，目前演示合照检测 10 张人脸、自动匹配 10 人，召回率和精确率均为 1.0。`scripts/prepare_demo.py` 还会额外生成 50 人压力合照 `demo_collage_50_pressure.png`，可用 `scripts/group_collage_50_selftest.py` 作为“可处理 10–50 人多人合照”的加分证据。合照页面还提供“自动识别 + 教师人工确认/补选”闭环，更符合真实系统。

## 6. 目录结构

```text
attendance_system/
  app.py                  # Flask 入口与 API 路由
  core/
    config.py             # 路径和阈值配置
    db.py                 # SQLite 初始化、种子账号、审计日志
    security.py           # 登录态、角色权限
    vision.py             # 人脸检测/特征/活体/情绪/合照标注
  templates/index.html    # 单页前端
  static/css/app.css      # 页面样式
  static/js/app.js        # 前端交互与摄像头采集
  data/app.db             # 首次运行自动生成
  storage/                # 上传图、样本图、合照标注图
  docs/                   # 报告与说明文档
  tests/                  # 后续测试脚本
```

## 7. 隐私与数据合规

- 不内置任何真实人脸数据集。
- 所有人脸样本仅保存在本地 `storage/faces`，数据库保存相对路径和特征向量。
- 课程验收结束后可删除 `data/app.db` 与 `storage/` 清理个人数据。

## 8. 一键验收命令

```powershell
python scripts\prepare_demo.py
python -m compileall .
python scripts\smoke_test.py
python scripts\group_collage_selftest.py
python scripts\group_collage_50_selftest.py
python scripts\final_acceptance.py
python app.py
```

关键报告：

- `docs/face_data_import_report.json`：班级照片导入覆盖率；
- `docs/face_data_evaluation_report.json`：阈值与误识别率评估；
- `docs/demo_collage_report.json`：10 人演示合照名单；
- `docs/demo_collage_50_report.json`：50 人压力合照名单；
- `docs/group_collage_selftest_report.json`：合照接口自测准确率；
- `docs/group_collage_50_selftest_report.json`：50 人合照压力测试结果；
- `docs/现场验收展示清单.md`：现场逐分展示路线。
