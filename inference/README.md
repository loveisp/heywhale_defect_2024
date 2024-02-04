## 下载模型权重

从百度网盘：https://pan.baidu.com/s/16atnFSjTkFN4BrWlEAE5lg?pwd=ahnw 下载模型权重，将下载的models目录放到inference目录下。

## 目录结构

整个目录结构如下：
- inference
    - data
        - test_B
            - csv文件
            - imgs
            - masks
            - 提交样例.csv
    - models
        - convnext_base
            - model_0_0.00020892962347716094.pth
            - model_3_0.0002511886414140463.pth
        - convnext_large
            - model_0_0.00036307806149125097.pth
            - model_2_0.00020892962347716094.pth
        - maxvit_t
            - model_1_0.00043651582673192023.pth
            - model_3_0.00043651582673192023.pth
        - vit_h_14
            - model_0_4.365158383734525e-06.pth
            - model_2_5.248074739938602e-06.pth
            - model_3_7.585775892948732e-06.pth
    - src
    - outputs
    - README.md
    - requirements.txt

## 运行推理代码

在inference目录下，运行：python ./src/run.py

运行之后，会在outputs目录下生成preds_testB.csv，即是我的B榜分数0.7926的提交文件。

整个推理过程在我这边的单卡4090上大概需要7分钟左右。

如果显存爆了，则需要把d_info字典中每个模型的bs改小一些，相应地，推理时间也会更久。

## 运行环境

可参考requirements.txt，都是非常常用的包。