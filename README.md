# Fully_Convolutional_Network_Time_Series
使用全卷积神经网络处理时序数据
<<<<<<< HEAD

# 《Time Series Classification from Scratch with Deep Netural Networks: A Strong Baseline》论文总结

## 综述：
在论文中，主要论述了使用新的三种方法来对时序数据进行分类，这三种方法均是基于深度学习，与原来的distance-based基于距离和feature-based基于特征的方法不同，深度学习思想的方法在实验结果、模型性能等方面较前二者相比都有着较大的提升。

这三种方法分别是Multilayer Perceptrons多层感知机、Fully Ｃonvolutional Networks 全卷积神经网络、Residual Network 残差网络。

---
### 多层感知机

论文中，此模型包括三个全连接层，每层上包含500个神经元，同时在多层感知机结构的基础上，还设置了Dropout和Relu激活函数。Dropout设置的作用是能够防止深度较大的模型在非常小的数据集上面产生过拟合现象，而Relu激活函数的设置，能够使模型在很深的情况下防止出现梯度饱和的问题。

### 全卷积神经网络
全卷积神经网络在图片分割方面展现出非常好的性能。在本论文中，全卷积神经网络被用作特征提取器，最终的输出仍然需要经过Softmax层得出结果。全卷积神经网络基本的模块构成为——卷积层、Batch_Normalization层、Relu激活函数层，其中卷积层使用一维卷积实现。那么对于这样的三者合一的基本模块，全卷积神经网络总共包含三个，其中卷积层的卷积核个数分别为：128、256、128，除此之外，全卷积网络模型不包含池化层，仅仅在最后输出时将数据再放入全局池化层。这样的做法也在后一种模型（残差网络）中使用。

BN层的设置，能够加快模型拟合数据的收敛速度，同时改善模型的泛化能力，在三层全卷积快之后，紧接着是全局池化层，而不是传统意义的全连接层，这样的设置能够极大的减少参数的个数。模型的最后一层仍然是softmax层。

### 残差网络

参差网络通过在每一个残差块之间添加快捷连接（shortcut），将神经网络扩展成更加深的结构，这样能够使求解的梯度流直接通过网络底层。其在目标检测和其他视觉相关的任务中均表现出非常好的效果。残差网络的基本模块是在卷积网络的基础上构造的，比如说一个Res_Block便是由卷积层+BN+Relu这样组合构成，该论文中每三个这样的组合便组成了一个残差块，共有三个残差块，其中卷积核的个数分别为64、128、128，最后两层仍然是全局池化和SoftMax层。

---

2019.08.30更
### 为什么全卷积神经网络可以处理时序数据？

读完上一篇论文之后，我仍然没有想明白，为什么以卷积为基础的神经网络可以处理时序数据，并且处理的效果还要好于曾经的霸主——基于RNN的各种神经网络。

在上一篇论文中，论文作者并没有从原理上直接解释清楚原因，而是进行大量的实验，分别使用卷积和rnn这两种模型为基础，训练UCR数据，通过对实验结果进行对比，大概有90%的实验结果显示，卷积的效果要好于后者。

带着这样的疑问，我又相继看了两篇论文，分别是：
《LSTM Fully Convolutional Networks for Time Series Classification》、《Insights into LSTM Fully Convolutional Networks for Time Series Classification》。以下我将分别介绍两篇论文。
___

### LSTM Fully Convolutional Networks for Time Series Classification
在这篇论文中，作者在全卷积的基础上增设了LSTM模块，属于对原有的全卷积模型的一种增强。下面是LSTM-FCN模型结构图：

![LSTM-FCN](./images/LSTM-FCN.png)

忽略一些细节，我们可以看到：
* 同一个输入分别输入给了卷积神经网络和LSTM神经网络
* 经过两个网络处理后的数据最终会Concat，然后输入给softmax层，得出分类概率

在这篇论文的Background部分，除了千篇一律的对RNN、LSTM、CNN的原理介绍，我还发现了令我眼前一亮的东西：*“Temporal Convolutions”*，我把它翻译为 *“时刻卷积”*

文中介绍到，对于时刻卷积网络的输入通常是时间序列数据。顺藤摸瓜，我找到了介绍 *Temporal Convolutions*的论文：《Temporal Convolutional Networks: A Unified Approach To Action Segmentation》,第一页长这样：

![TCN](./images/TCN.png)
这篇16年的论文，首次提出了使用卷积神经网络的思想去处理时序数据。从右图可以得出，此模型包括两个模块。Encoder、Decoder。

对于Encoder,主要就是应用了Conv、Pool、Normalize。文中有这样的一句话：*“For each of the L convolutional layers in the encoder, we apply a set of 1D filters that capture how the input signals evolve over the course of an action.”*
那么这句话很可就是我需要找的东西。我对它的理解是：
*“在编码器模块中，每一个卷积层通过设置一系列的一维卷积核，能够捕获一帧中的输入数据是怎么变化的”*

好，那我们看看具体是怎样的？
* 首先每一层的卷积核都将通过权重张量$W_i$、偏置张量$b_i$ 初始化，那么时序数据是如何通过Encoder中的每一层的呢？

$$
E_{i,t}^{(l)} = f(b_i^{(l)}+\sum_{t^{'}=1}^d<W_{i,t^{'}{,}}{E_{t+d-t^{'}}^{(l-1)}}>)
$$

这个数学公式就是最好的解释，简单说一下参数：

$l$：*index of layers*

$d$：*the duration of filters*

$E$: Acti