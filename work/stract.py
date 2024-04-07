##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.25  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.copy(fileDir + name, tarDir + name)
    return


if __name__ == '__main__':
    fileDir = "C:/Users/yyt/Desktop/dataset/100%/train/neg/"  # 源图片文件夹路径
    tarDir = 'C:/Users/yyt/Desktop/dataset/25%/neg/'  # 移动到新的文件夹路径
    moveFile(fileDir, tarDir)
















