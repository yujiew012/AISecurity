# AI Security — AI 安全知识库

## 概述

本项目用于收集与整理 **AI 安全** 相关的论文、工具与资源，支持团队协作与个人学习空间。

## 目录结构

```
.
├── Individual/   # 个人空间：自建目录存放个人资料、笔记、暂存（无需审核）
├── Tools/        # 工具库：AI 安全相关工具、代码、数据集、测试框架（须审核）
└── Papers/       # 论文库：按主题分类的论文 PDF（须审核）
```

各目录详细说明见：[Individual](Individual/README.md) | [Tools](Tools/README.md) | [Papers](Papers/README.md)

## 协作与 Git 使用

依据团队协作规范，通过 **Pull Request (PR)** 向本仓库贡献内容。根据你是否有仓库写权限，选择以下两种方式之一。

> **注意**：无论哪种方式，请勿直接在 `master` 上修改提交，先拉取最新代码后**新建分支**再操作，保持 `master` 干净。

---

### 方式一：直接 Clone（有仓库写权限）

适用于已被添加为仓库 **Collaborator** 的成员。

#### 1. 首次参与：Clone 仓库

```bash
git clone https://github.com/<仓库地址>/AISecurity.git
cd AISecurity
```

#### 2. 日常使用：同步与提交

```bash
# 拉取最新代码
git pull origin master

# 新建分支，避免污染 master
git checkout -b your-branch

# 修改完成后，暂存并提交
git add Individual/your-name/
git commit -m "docs: 添加 xxx 笔记/工具/论文"

# 推送分支到远程
git push origin your-branch
```

#### 3. 发起 PR

在 GitHub 仓库页面发起 PR，将 `your-branch` 合并到 `master`（详见下方 [如何发起 PR](#如何发起-pull-request)）。

---

### 方式二：Fork（无仓库写权限）

适用于没有仓库写权限的外部贡献者或普通成员。

#### 1. 首次参与：Fork + Clone

```bash
# 在 GitHub 上点击 "Fork" 将仓库复制到你的账号下

# 克隆你 Fork 后的仓库
git clone https://github.com/<your-username>/AISecurity.git
cd AISecurity

# 添加上游仓库（即主仓库），便于后续同步
git remote add upstream https://github.com/<原仓库地址>/AISecurity.git
```

#### 2. 日常使用：同步与提交

```bash
# 从上游拉取最新代码，保持 master 与主仓库一致
git pull upstream master

# 新建分支，避免污染 master
git checkout -b your-branch

# 修改完成后，暂存并提交
git add Individual/your-name/
git commit -m "docs: 添加 xxx 笔记/工具/论文"

# 推送分支到你的 Fork（origin）
git push origin your-branch
```

#### 3. 发起 PR

在 GitHub 上从**你的 Fork** 向**主仓库**的 `master` 发起 PR（详见下方 [如何发起 PR](#如何发起-pull-request)）。

---

### 如何发起 Pull Request

推送分支后，在 GitHub 上创建 PR，将你的分支合并到 `master`：

1. 打开 GitHub 仓库页面（方式二则打开主仓库页面），你会看到顶部出现一条黄色提示：**"your-branch had recent pushes — Compare & pull request"**，点击该按钮。
   - 如未看到提示，也可手动操作：点击 **"Pull requests"** 标签页 → **"New pull request"** → 将 **base** 设为 `master`，**compare** 设为你的分支名。
2. 填写 PR 标题和描述，简要说明本次修改内容。
3. 点击 **"Create pull request"** 提交。
4. 等待审核通过后，由维护者点击 **"Merge pull request"** 合并到 `master`。

### 常用命令速查

| 场景           | 命令 |
|----------------|------|
| 查看远程       | `git remote -v` |
| 拉取最新（方式一） | `git pull origin master` |
| 拉取最新（方式二） | `git pull upstream master` |
| 查看当前分支   | `git branch` |
| 新建并切换分支 | `git checkout -b 新分支名` |
| 切回 master    | `git checkout master` |
| 查看修改       | `git status` / `git diff` |
| 撤销未暂存修改 | `git checkout -- <文件>` |
| 查看提交历史   | `git log --oneline -10` |

### 提交到 Papers / Tools 的流程

1. 在 `Individual/<name>/` 下整理好内容（论文 PDF、工具说明等）。
2. 按 [Papers](Papers/README.md) 或 [Tools](Tools/README.md) 的收录要求自检。
3. 通过上述 Git 流程提交并创建 PR，等待审核。
4. 审核通过后由维护者合并；通过审核的内容可放入 `Papers/` 或 `Tools/`。
