# GitHub Automation Setup Instructions

完整的自动化设置说明，包括数据上传和 Google Drive 配置。

## 目录

1. [打包并上传地理数据到 GitHub Releases](#1-打包并上传地理数据到-github-releases)
2. [配置 Google Drive API](#2-配置-google-drive-api)
3. [配置 GitHub Secrets](#3-配置-github-secrets)
4. [更新工作流程文件](#4-更新工作流程文件)
5. [测试自动化](#5-测试自动化)

---

## 1. 打包并上传地理数据到 GitHub Releases

### 步骤 1.1: 运行打包脚本

```bash
cd "/Users/iris/Dropbox/sandiego code/code/fieldprep"
./scripts/package_data.sh
```

这将创建 `geospatial_data.tar.gz` 文件（大约 50-60 MB）。

需要打包的文件夹（共6个）:
- `sd_paving_segs_datasd/` (25M) - 街道数据
- `addrapn_datasd/` (89M) - 地址数据
- `council_districts_datasd/` (696K) - 市议会区
- `zoning_datasd/` (6.9M) - 分区数据
- `cmty_plan_datasd/` (1.5M) - 社区规划
- `san_diego_boundary_datasd/` (544K) - 城市边界

### 步骤 1.2: 上传到 GitHub Releases

1. 进入你的 GitHub 仓库页面
2. 点击 **"Releases"** → **"Create a new release"**
3. 填写发布信息：
   - **Tag version**: `v1.0-data`
   - **Release title**: `Geospatial Data v1.0`
   - **Description**:
     ```
     Required geospatial data files for weekly plan generation.

     Contains:
     - Street segments
     - Address points
     - Council districts
     - Zoning
     - Community plan areas
     - City boundary
     ```
4. 上传文件：点击 "Attach binaries" → 选择 `geospatial_data.tar.gz`
5. 点击 **"Publish release"**

### 步骤 1.3: 获取下载链接并更新工作流程

发布后，你会得到类似这样的下载链接：
```
https://github.com/IrisWang2333/fieldprep/releases/download/v1.0-data/geospatial_data.tar.gz
```

**重要**: 确认 `.github/workflows/weekly-plan-emit.yml` 文件中的 Release URL 已使用正确的仓库名 `IrisWang2333/fieldprep`。

---

## 2. 配置 Google Drive API（OAuth 方式）

> **注意**：当前工作流使用 **OAuth2 凭证**，而非 Service Account。以下步骤基于此方式。

### 步骤 2.1: 创建 Google Cloud 项目

1. 访问 [Google Cloud Console](https://console.cloud.google.com/)
2. 创建新项目或选择现有项目
3. 项目名称可以是: `fieldprep-automation`

### 步骤 2.2: 启用 Google Drive API

1. 在项目中，进入 **"APIs & Services"** → **"Library"**
2. 搜索 **"Google Drive API"**
3. 点击 **"Enable"**

### 步骤 2.3: 创建 OAuth2 凭证

1. 进入 **"APIs & Services"** → **"Credentials"**
2. 点击 **"Create Credentials"** → **"OAuth client ID"**
3. 应用类型选择 **"Desktop app"**
4. 填写名称（如 `fieldprep-uploader`）
5. 点击 **"Create"** → 下载 JSON 凭证文件

### 步骤 2.4: 生成 OAuth Token

1. 使用下载的 JSON 文件在本地运行认证流程，生成含 access_token/refresh_token 的凭证文件
2. **保存好这个凭证 JSON 文件**，需要用它来配置 GitHub Secrets

---

## 3. 配置 GitHub Secrets

### 步骤 3.1: 添加 GOOGLE_DRIVE_OAUTH_CREDENTIALS

1. 进入 GitHub 仓库
2. 点击 **"Settings"** → **"Secrets and variables"** → **"Actions"**
3. 点击 **"New repository secret"**
4. 填写：
   - **Name**: `GOOGLE_DRIVE_OAUTH_CREDENTIALS`
   - **Secret**: 粘贴步骤 2.4 生成的 OAuth 凭证 JSON 内容（包含 access_token 和 refresh_token）
5. 点击 **"Add secret"**

### 步骤 3.2: 添加 GOOGLE_DRIVE_FOLDER_ID

1. 点击 **"New repository secret"**
2. 填写：
   - **Name**: `GOOGLE_DRIVE_FOLDER_ID`
   - **Secret**: `17Eexa-x7fOIB0gOu63SWUkZlNSr5oyk8`（field files 文件夹）
3. 点击 **"Add secret"**

### 步骤 3.3: 添加 GOOGLE_DRIVE_ROUTING_FOLDER_ID

1. 点击 **"New repository secret"**
2. 填写：
   - **Name**: `GOOGLE_DRIVE_ROUTING_FOLDER_ID`
   - **Secret**: routing files 对应的 Google Drive 文件夹 ID
3. 点击 **"Add secret"**

---

## 4. 更新工作流程文件

确保 `.github/workflows/weekly-plan-emit.yml` 文件中的 GitHub 用户名和仓库名已正确更新：

```yaml
# 第 41 行附近
RELEASE_URL="https://github.com/IrisWang2333/fieldprep/releases/download/v1.0-data/geospatial_data.tar.gz"
```

提交更改：
```bash
cd "/Users/iris/Dropbox/SanDiego311/code/fieldprep"
git add .github/workflows/weekly-plan-emit.yml
git commit -m "Update workflow with correct GitHub repository URL"
git push
```

---

## 5. 测试自动化

### 步骤 5.1: 手动触发工作流程

1. 进入 GitHub 仓库
2. 点击 **"Actions"** 标签
3. 选择 **"Weekly Plan and Emit"** 工作流程
4. 点击 **"Run workflow"** → **"Run workflow"**

### 步骤 5.2: 检查运行日志

1. 等待工作流程开始运行（可能需要几秒钟）
2. 点击运行记录查看详细日志
3. 检查每个步骤是否成功：
   - ✓ Download geospatial data from GitHub Releases
   - ✓ Download latest notification activities
   - ✓ Calculate next Saturday date
   - ✓ Run plan.py
   - ✓ Run emit.py
   - ✓ Upload to Google Drive

### 步骤 5.3: 验证上传

1. 检查 Google Drive 文件夹: https://drive.google.com/drive/u/2/folders/17Eexa-x7fOIB0gOu63SWUkZlNSr5oyk8
2. 应该看到新上传的 CSV 文件（以日期命名的文件夹）

---

## 自动运行时间

工作流程配置为每周五晚上 10:00 PM（美西时间）自动运行。

- **Cron 表达式**: `0 6 * * 6`（UTC 时间周六早上 6:00）
- **等同于**: 美西时间周五晚上 10:00 PM（PST）

---

## 故障排查

### 问题 1: 下载地理数据失败

**错误**: `404 Not Found` 下载 geospatial_data.tar.gz

**解决方案**:
- 确认 GitHub Release 已经发布
- 检查 `.github/workflows/weekly-plan-emit.yml` 中的 URL 是否正确
- 确认 tag 是 `v1.0-data`

### 问题 2: Google Drive 上传失败

**错误**: `403 Forbidden` 或 `HttpError 403`

**解决方案**:
- 确认 `GOOGLE_DRIVE_OAUTH_CREDENTIALS` secret 内容正确（包含 access_token 和 refresh_token）
- 确认 `GOOGLE_DRIVE_FOLDER_ID` 和 `GOOGLE_DRIVE_ROUTING_FOLDER_ID` secrets 已设置
- 检查 OAuth token 是否过期，必要时重新生成并更新 secret

### 问题 3: 路径错误

**错误**: `No such file or directory: /Users/iris/...`

**解决方案**:
- 检查 `utils/data_fetcher.py` 中的 `get_project_root()` 函数
- 确认使用相对路径而非绝对路径
- 检查 `config/datasources.yaml` 中的路径配置

---

## 完成检查清单

- [ ] 运行 `./scripts/package_data.sh` 成功
- [ ] 上传 `geospatial_data.tar.gz` 到 GitHub Releases (tag: v1.0-data)
- [ ] 创建 Google Cloud 项目并启用 Drive API
- [ ] 创建 OAuth2 凭证并生成含 token 的凭证文件
- [ ] 添加 GitHub Secrets: `GOOGLE_DRIVE_OAUTH_CREDENTIALS`、`GOOGLE_DRIVE_FOLDER_ID`、`GOOGLE_DRIVE_ROUTING_FOLDER_ID`
- [ ] 更新 `.github/workflows/weekly-plan-emit.yml` 中的 GitHub 仓库 URL
- [ ] 手动触发工作流程测试成功
- [ ] 验证文件已上传到 Google Drive

---

**完成所有步骤后，自动化就配置好了！每周五晚上 10:00 PM 会自动运行并上传结果到 Google Drive。**
