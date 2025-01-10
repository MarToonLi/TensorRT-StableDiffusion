# CNSD项目使用流程
基于tong:v2镜像执行以下内容：
tong:v1镜像中的内容是修复了本地下无法执行的问题；
tong:v2镜像修复了sshd连接、重启自启动的问题；

## 一 CNSD项目安装
```shell
# step1. 切换到 player 用户
su player

# step2. 如果想一直使用 root 用户，需要安装tensorrt python 包
pip install /home/player/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl

# step3. 测试跑原生torch diffusion 代码+联网下载模型
cd /home/player/ControlNet/
python3 compute_score.py 

# step4. 下载项目代码
# 必须下载，此代码为在原生代码基础上进行了一些完善，会好用一些。
cd /home/player
git clone https://github.com/shenlan2017/TensorRT-StableDiffusion.git

# step5. 切换到 export_onnx 分支
cd TensorRT-StableDiffusion
git checkout export_onnx

# step6. copy 模型，修改代码中模型路径（ldm/modules/encoders/modules.py 100行左右）
# 模型文件解压到x://ex_space目录下；文件将同步映射到容器的/data文件夹中。
# 构建成如下结构：X:\ex_space\diffusion_models（cache和models）
cd /home/player/TensorRT-StableDiffusion  # add
cp -rv /data/ex_space/diffusion_models/models .
cp -rv /data/ex_space/diffusion_models/cache/*   /home/player/.cache/huggingface
mkdir openai
cd openai
cp /data/ex_space/diffusion_models/clip-vit-large-patch14.tar.gz . &&  tar -zxvf clip-vit-large-patch14.tar.gz -C clip-vit-large-patch14 && rm models.tar  # add



# 跑torch 版本测试
cd /home/player/TensorRT-StableDiffusion
python3 compute_score_torch.py
```

## 二 配置python执行环境

```shell
# anaconda
wget -O /opt/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh "https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh"
chmod +x /opt/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh
sh -c '/bin/echo -e "\nyes\n\nyes" | sh /opt/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -b -p /opt/miniconda3' 

# 配置环境变量
echo 'export PATH=/opt/miniconda3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 配置访问源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
 && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
 && conda config --set show_channel_urls yes \
 && conda config --set auto_activate_base no  \
 && conda config --set ssl_verify false

# 若粘贴复制执行，则需要手动敲y
conda create --name general python=3.8.10   
conda update --name base conda 
conda init bash 
source activate 
conda activate general 
conda install -y --quiet numpy pyyaml mkl mkl-include setuptools cmake cffi typing 
conda install -y --quiet -c mingfeima mkldnn 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**tong:v3镜像**
**docker commit trt2025_v2 tong:v3 -m "lfs and diffusion_models are commit." -a "martoonli"**
**docker run -dit -v X:/ex_space:/data/ex_space  -v F:/Projects:/data/Projects  -p 6677:22  --user root --gpus all --name trt2025_v3 tong:v3**
**今后，trt2025_v3容器会在启动后，直接运行TRT—CNSD项目、等项目**
**driver：537; cuda: 12.2; cudnn: **


## 三 配置自己的项目
cd /data/projects/TensorRT-StableDiffusion
1. **需要执行enable_torch.sh或者enable_trt.sh文件**
2. python compute_score_torch.py















