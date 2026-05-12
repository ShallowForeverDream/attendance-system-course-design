# 本目录用于放置老师授权的人脸照片

为了保护同学人脸与个人信息，GitHub 仓库不会提交真实 `face_data`。

组员本地复现时，请把老师提供的照片目录放到项目同级目录，例如：

```text
实验六-课程设计/
  face_data/
    学号-姓名-专业-性别.jpg
  attendance_system/
```

然后在 `attendance_system` 中运行：

```powershell
python scripts\prepare_demo.py --source ..\face_data
```

脚本会自动生成本地数据库、样本裁剪图、演示合照和自测报告。
