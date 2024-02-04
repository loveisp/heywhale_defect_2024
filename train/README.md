## 目录结构

整个目录结构如下：
- train
    - data
        - train
            - csv文件
            - 文件标签汇总数据.csv
        - test_A
            - csv文件
            - 提交样例.csv
    - src
        - extract.py
        - train.py
    - README.md
    - requirements.txt
    
## 提取图像

在train目录下，运行：python ./src/extract.py

运行之后，会根据data目录中的train和test_A的csv文件，在相应目录中生成imgs和masks两个目录。

可以根据机器的cpu数量调整进程数，即extract.py中的processes变量。

我在本地机器把进程数设为16，整个提取图像的过程大约需要半分钟。

## 运行训练代码

在train目录下，运行以下命令：
- python ./src/train_convnext_large.py 0
- python ./src/train_convnext_large.py 2
- python ./src/train_convnext_base.py 0
- python ./src/train_convnext_base.py 3
- python ./src/train_maxvit_t.py 1
- python ./src/train_maxvit_t.py 3
- python ./src/train_vit_h_14.py 0
- python ./src/train_vit_h_14.py 2
- python ./src/train_vit_h_14.py 3

也可以直接执行脚本文件：bash ./src/train.sh，脚本文件会依次执行上面的命令。

运行之后，会在models目录里生成多个模型文件，即为我上传到百度网盘：https://pan.baidu.com/s/16atnFSjTkFN4BrWlEAE5lg?pwd=ahnw 的用于推断的模型文件。

同时，也会在logs目录里生成日志记录。

如果显存爆了，请把get_ds函数中的bs参数调小。

## 运行环境

可参考requirements.txt，除fastai外都是非常常用的包。