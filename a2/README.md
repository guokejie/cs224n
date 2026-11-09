# 欢迎参加作业 2！

本次作业将使用 PyTorch。如果你还不熟悉 PyTorch，或者想复习一下 PyTorch 的基础知识，可以在 Canvas 的 **Course Videos（课程视频）** 中观看 PyTorch 复习课程。

如果你想继续使用作业 1 中的 `cs224n` 环境，请确认已安装 `local_env.yml` 中列出的所有依赖。请运行：

```bash
# 1. 激活旧环境
conda activate cs224n

# 2. 安装 docopt
conda install docopt

# 3. 安装 pytorch、torchvision 和 tqdm
conda install pytorch torchvision -c pytorch
conda install -c anaconda tqdm
```

如果你想为本次作业新建一个环境，请运行：

```bash
# 1. 根据 local_env.yml 中指定的依赖创建环境
# （根据电脑配置不同，此过程可能需要一些时间）
conda env create -f local_env.yml

# 2. 激活新环境
conda activate cs224n_a2
```

停用当前环境时，请运行：

```bash
conda deactivate
```