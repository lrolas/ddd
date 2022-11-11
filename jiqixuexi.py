from datetime import time

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist     #tf.keras.datasets是一个公开的API可以直接将数据load
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()   #1从给定地址下载数据集 2处理下载下来的四个文件
print("训练集",train_images.shape)     #3维数组 ，三个数分别为高，宽，通道数
print("训练集标签",train_labels.shape)  #一维数组  ，一个数为几个元素
print("测试集",test_images.shape)      #3维数组   三个数分别为高，宽，通道数
print("测试集标签 ",len(test_labels))   #一维数组   一个数为几个元素

# print(train_images)
# 训练集(60000, 28, 28)
# 训练集标签 (60000,)
# 测试集(10000, 28, 28)
# 测试集标签 10000

#指定名称
"""

1 T恤
2 裤子
3 套头衫
4 连衣裙
5 外套
6 凉鞋
7 衬衫
8 运动鞋
9 包
10 短靴
"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#绘图
plt.figure()
plt.imshow(train_images[0])

plt.grid(1)  # #显示网格线
plt.show()


#归一化（将图像矩阵转化到0-1之间，为了保证精度，经过了运算的图像矩阵I其数据类型会从unit8型变成double型。图像时对double型是认为在0 ~ 1范围内，uint8型时是0~255范围）
train_images = train_images / 255.0

test_images = test_images / 255.0
#图像归一化是计算机视觉、模式识别等领域广泛使用的一种技术。所谓图像归一化, 就是通过一系列变换, 将待处理的原始图像转换成相应的唯一标准形式(该标准形式图像对平移、旋转、缩放等仿射变换具有不变特性)。
# 归一化， 其基本工作原理为: 首先利用图像中对仿射变换具有不变性的矩来确定变换函数的参数,
# 然后利用此参数确定的变换函数把原始图像变换为一个标准形式的图像(该图像与仿射变换无关)。  一般说来, 基于矩的图像归一化过程包括 4 个步骤 即坐标中心化、x-shearing 归一化、缩放归一化和旋转归一化。
# 基本上归一化思想是利用图像的不变矩寻找一组参数使其能够消除其他变换函数对图像变换的影响。也就是转换成唯一的标准形式以抵抗仿射变换。图像归一化使得图像可以抵抗几何变换的攻击
#，它能够找出图像中的那些不变量，从而得知这些图像原本就是一样的或者一个系列的。以下你要知道的：
#1.归一化处理并没有改变图像的对比度
#2.归一化处理很简单，假设原图像是8位灰度图像，那么读入的像素矩阵最大值为256，最小值为1，定义矩阵为I，J＝I／256，就是归一化的图像矩阵，就是说归一化之后所有的像素值都在［0，1］区间内。


#建立模型
model = tf.keras.Sequential()   #建立模型


#主要的功能就是将(28,28)像素的图像即对应的2维的数组转成28*28=784的一维的数组.
#model.add 添加层
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #展平层 用来对数组进行展平操作的
model.add(tf.keras.layers.Dense(64,activation='relu'))  #全连接层 指定该网络层中的神经元个数为64，activation 表示激活函数，以字符串的形式给出，包括relu、softmax、Sigmoid、Tanh 等
                                                        #activation = 'relu'的意思是我们将激活函数设定为relu。简单地说，我们告诉这个层转换所有的负值为0。
model.add(tf.keras.layers.Dropout(0.5))                 #这里在输入层之后添加一个新的Dropout层，Dropout率设为50%——即每个更新周期中50%的输入将被随机排除。
model.add(tf.keras.layers.Dense(10,activation='softmax'))  #全连接层 指定该网络层中的神经元个数为64，activation='softmax'
                                                          # 表示激活函数，常用且重要的一种归一化函数，其将输入值映射为0-1之间的概率实数，常用于多分类。
model.summary()         #model.summary()输出模型各层的参数状况

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#参数 loss 是损失函数，optimizer 是优化器，metrics 是模型训练时，我们希望输出的评测指标。
#Adam:对RMSProp优化器的更新.利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率. 优点:每一次迭代学习率都有一个明确的范围,使得参数变化很平稳.


#训练的函数
# epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)   #输入数据和标签,输出损失和精确度.
print('\nTest accuracy:', test_acc)         #输出精确度
model.save("model/model1.h5")   #保存模型

#编译模型
#Adam方法，此处可选。 损失函数是SparseCategoricalCrossentropy
#参数 loss 是损失函数，optimizer 是优化器，metrics 是模型训练时，我们希望输出的评测指标。
#Adam:对RMSProp优化器的更新.利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率. 优点:每一次迭代学习率都有一个明确的范围,使得参数变化很平稳.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#开始训练
train_images=train_images.reshape(60000,28,28,1)
hist = model.fit(train_images, train_labels, epochs=30)   #fit()训练函数  model.fit( )函数返回一个History的对象，即记录了loss和其他指标的数值随epoch变化的情况。
# train_images 输入的数据
# train_labels 输入的标签
# epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch


#模型评估
test_images=test_images.reshape(10000,28,28,1)  #升维，10000图片个数，28×28是图片矩阵大小，当图像为灰度时，只需要1个 channel 所以它会是 60000x28x28 x1
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)  # 直接得到最后的测试集准确率

print('\nTest accuracy:', test_acc)


#图形测试定义
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])  #归一化

predictions = probability_model.predict(test_images)                         #预测

#print(predictions.shape)
#显示图片，正确的标签是绿色，错误的是红色
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)  #plt.grid() # 显示网格线 1=True=默认显示；0=False=不显示
  plt.xticks([])   #获取或设置当前x轴刻度位置和标签。若不传递任何参数，则返回当前刻度值
  plt.yticks([])    #获取或设置当前y轴刻度位置和标签。若不传递任何参数，则返回当前刻度值

  plt.imshow(img, cmap=plt.cm.binary)  #img是获取的图像数据 #cmap:此参数是颜色图实例或注册的颜色图名称

  predicted_label = np.argmax(predictions_array)   #np.argmax 这是个二维数组 (1）遵循运算之后降一维的原则，因此返回的会是一个一维的array。（2）函数返回的是最大值的索引，而不是最大值本身。
  if predicted_label == true_label:  #预测正确,颜色为绿
    color = 'green'
  else:                              #预测错误，颜色为红
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],   #xlabel设置刻度
                                100*np.max(predictions_array),         #format用于遍历
                                class_names[true_label]),
                                color=color)


#定义一个函数来显示图片每个概率的柱状图，正确的是绿色，错误的是红色
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)        #无网格
  plt.xticks(range(10))   #x轴有10个刻度
  plt.yticks([])          #获取或设置当前y轴刻度位置和标签。若不传递任何参数，则返回当前刻度值
  thisplot = plt.bar(range(10), predictions_array, color="#777777")  #柱状图，上传图片，指定x轴上数值，颜色为灰色
  plt.ylim([0, 1])     # plt.ylim函数功能:设置x轴ly轴的数值显示范围。
  predicted_label = np.argmax(predictions_array)    #np.argmax 这是个二维数组 (1）遵循运算之后降一维的原则，因此返回的会是一个一维的array。（2）函数返回的是最大值的索引，而不是最大值本身。

  thisplot[predicted_label].set_color('red')   #错误设置红色
  thisplot[true_label].set_color('green')      #正确设置绿色

#预测多张结果
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols  #15张图片
plt.figure(figsize=(2*2*num_cols, 2*num_rows))  #figsize:指定figure的宽和高，单位为英寸
for i in range(num_images):   # for i in range () 就是给i赋值
  plt.subplot(num_rows, 2*num_cols, 2*i+1)   #第一个参数代表子图的行数；第二个参数代表该行图像的列数； 第三个参数代表每行的第几个图像。
  plot_image(i, predictions[i], test_labels, test_images)  #调用函数plot_image
  plt.subplot(num_rows, 2*num_cols, 2*i+2)                 #第一个参数代表子图的行数；第二个参数代表该行图像的列数； 第三个参数代表每行的第几个图像。
  plot_value_array(i, predictions[i], test_labels)         #调用函数plot_value_array
plt.tight_layout()                                         #tight_layout会自动调整子图参数，使之填充整个图像区域。
plt.show()

