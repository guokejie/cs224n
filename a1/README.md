# 欢迎来到 CS224N！

本课程将全程使用 Python。如果你已经配置好了 Python 环境，那就太好了！但请确保 Python 版本至少为 **3.8**。

如果还没有配置，最简单的方法是先确保电脑上至少有 **3GB 可用空间**，然后前往 [Anaconda 下载页面](https://www.anaconda.com/download/)，下载并安装 Python 3 版本的 Anaconda。它支持各类操作系统。

安装 conda 后，请关闭所有已打开的终端窗口。然后打开一个新终端，运行以下命令。

## 1. 根据 env.yml 中指定的依赖创建环境

```bash
conda env create -f env.yml
```

## 2. 激活新环境

```bash
conda activate cs224n
```

## 3. 在新环境中安装 IPython 内核，以便在 Jupyter Notebook 中使用该环境

```bash
python -m ipykernel install --user --name cs224n
```

## 4. 启动作业 1 的 Jupyter Notebook

只有作业 1 使用 Jupyter Notebook。完成上述步骤后，输入以下命令即可开始：

```bash
jupyter notebook exploring_word_vectors.ipynb
```

## 5. 确认 Notebook 使用的是正确的环境

在 `exploring_word_vectors.ipynb` 的顶部菜单栏中，依次点击 **Kernel → Change kernel（内核 → 更换内核）**，然后在下拉菜单中选择 **cs224n**。

## 退出当前激活的环境

```bash
conda deactivate
```