# 运行说明

声明:本项目根据论文usad的开源代码改编，将Jupyter版本的主函数代码整合到一个.py文件，并给代码的重要部分进行了注释，方便阅读和运行原项目请看usad论文的原始开源项目，欢迎大家一起交流学习!

1 安装PyTorch 1.6.0，CUDA 10.1,Pandas,Seaborn,Sklearn,Matplotlib(to allow use of GPU, not compulsory)

2 修改gdrivedl.py中的下载网址和保存文件夹，然后运行两次，分别下载正常数据集和异常数据集，保存到input文件夹中(或者使用文件夹下已经处理好的数据集)

3 修改run.py中的相关文件路径后，运行run.py



注:注释版代码仅供阅读，真正能运行且效果比较好的代码在Right_Version
