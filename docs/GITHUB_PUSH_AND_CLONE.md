# GitHub 推送与组员复现步骤

## 1. 推送到 GitHub

本仓库已经完成本地 Git 初始化和首次提交。由于人脸数据属于课程授权数据，仓库不会提交：

- `face_data/` 原始照片；
- `data/app.db` 本地数据库；
- `storage/` 中生成的人脸裁剪图、合照和标注图；
- `docs/*.json` 本地验收报告。

如果当前机器已登录 GitHub CLI，可执行：

```powershell
gh repo create attendance-system-course-design --private --source . --remote origin --push
```

如果已经在 GitHub 网页上新建了仓库，则执行：

```powershell
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

## 2. 组员 clone 后复现

```powershell
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

把老师提供的 `face_data` 放到项目同级目录：

```text
实验六-课程设计/
  face_data/
  attendance_system/
```

然后运行：

```powershell
python scripts\prepare_demo.py --source ..\face_data
python -m compileall .
python scripts\smoke_test.py
python scripts\group_collage_selftest.py
python scripts\group_collage_50_selftest.py
python scripts\final_acceptance.py
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

默认账号：

| 角色 | 用户名 | 密码 |
|---|---|---|
| 教师 | `teacher` | `teacher123` |
| 学生 | `student01` | `student123` |
| 学生 | `student02` | `student123` |
