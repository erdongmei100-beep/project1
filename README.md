# project1
基于车载摄像头识别高速公路应急车道的占用情况

## 推送本地修改到 GitHub

在容器或本地环境中完成代码修改后，可按以下步骤将提交同步到你的 GitHub 仓库：

1. 查看当前分支：
   ```bash
   git branch --show-current
   ```
2. 如未配置远端，添加或更新 `origin` 地址：
   ```bash
   git remote add origin https://github.com/erdongmei100-beep/project1.git
   # 若已存在但需更新，可使用：
   # git remote set-url origin https://github.com/erdongmei100-beep/project1.git
   ```
3. 确认远端配置：
   ```bash
   git remote -v
   ```
4. （可选）启用 Git LFS 以支持大文件：
   ```bash
   git lfs install
   ```
5. 将当前分支推送到远端：
   ```bash
   git push -u origin HEAD
   ```

首次推送后，后续可直接使用 `git push` 同步最新提交。
