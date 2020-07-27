<h1 align="center">- MCNN论文复现 -</h1>

<p align="center">
<img src="https://img.shields.io/badge/version-2020.07.27-green.svg?longCache=true&style=for-the-badge">
<img src="https://img.shields.io/badge/license-GPL%20(%3E%3D%202)-blue.svg?longCache=true&style=for-the-badge">
</p>


<br/>
<br/>

<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/author.png"  alt="author" />

<br/>
<br/>

<p>
<a href=https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf>MCNN</a>是CVPR2016年的一篇论文, 作者提出了一种简单而有效的<b>多列卷积神经网络架构(Multi-column Convolutional Neural Network, MCNN)</b>，通过使用大小不同的卷积核去适应人/头部大小的变化，以将图片映射为人群密度图。作者在文章中使用了<b>几何自适应高斯核</b>去计算人群密度图(Ground Truth)，同时收集并标记了一个大型的新数据集(ShanghaiTech数据集)，其中包括1198幅图像，数据集可以在AI Studio上下载到: https://aistudio.baidu.com/aistudio/datasetdetail/10675 .
</p>


<br/>
<br/>

<p>
MCNN受MDNNs的启发，由三列并行的CNN组成，每列CNN卷积核大小不同。为了简化，所有列使用相同的网络结构(即conv-pooling-conv-pooling)。每次池化都会使用2*2的Max Pooling，而激活函数全部选择Relu。堆叠三列CNN的输出特征图，并使用1*1的卷积核将其映射为密度图。MCNN的整体架构图如<b>Figure 1</b>所示：
</p>

<br/>

<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/archit.png"  alt="archit" width="1000" height="400"/>

<p align="center"><b>图 1</b>：用于人群密度图估计的多列卷积神经网络(MCNN)的结构<br/>
<b>Figure 1</b>：The structure of the proposed multi-column convolutional neural network for crowd density map estimation.</p>
<br/>
<br/>

<p>
MCNN在训练时，存在<b>数据样本少和梯度消失</b>的问题，受预训练模型RBM的启发，作者将三列CNN单独进行预训练，将这些预训练的CNN参数初始化为对应的MCNN参数并微调。需要补充的是，MCNN使用了最简单的均方误差作为损失函数。
</p>

<br/>

<p>
论文中使用<b>几何自适应高斯核</b>去计算数据图片的Ground Truth：
</p>


<p align="center">
<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/formula.jpeg"  alt="公式" width="650" height="300"/>
</p>


<p>
在<b>Figure 2</b>中，显示了两张图片的人群密度图。值得说明的是，由于经过了两次下采样，所以预测出人群密度图的分辨率变为原来的1/4.
</p>

<p align="center">
<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/figure2.png"  alt="figure2"/>
</p>

<p align="center">
<b>图 2</b>：原始图像和通过几何自适应高斯核进行卷积获得的相应的人群密度图。</br>
<b>Figure 2</b>：Original images and corresponding crowd density maps obtained by convolving geometry-adaptive Gaussian kernels.
</p>


<p>
MCNN几乎可以从任何观察角度准确估计单个图像中的人群数，在2016年，取得了人群计数领域<b>state-of-art</b>的成绩。同时作者还指出，仅需要对模型最后几层进行微调，便可以将模型轻松迁移到目标问题，验证了模型的鲁棒性。</br></br>在论文中，还有很多细节，本篇不再赘述，可以查看原论文<a href=https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf>MCNN</a>
</p>

##### 1.基于<b>飞桨开源框架(Paddlepaddle)</b>复现MCNN

环境依赖:

```shell
paddlepaddle >= 1.7.0
numpy
matplotlib
PIL 
opencv-python
```

##### 2. 训练策略
	MCNN受到预训练模型的启发，先将三列CNN单独训练，之后将CNN的参数初始化为MCNN对应的参数之后，整体再进行训练，在原论文中，训练策略为批随机梯度下降法。

##### 3.模型复现效果

我们可以对比使用飞桨的训练效果和原论文的训练效果：

<p align="center">
<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/figure3.png"  alt="figure3"/>
</p>

<p align="center">
<b>图 3</b>：原论文中两张测试集图片的真实人群密度图和估计人群密度图</br>
<b>Figure 3</b>：The ground truth density map and estimated density map of our MCNN Model of the test image
</p>

<p align="center">
<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/figure4.png"  alt="figure4"/>
</p>

<p align="center">
<b>图 4</b>：基于飞桨的测试集图片回归结果，从左至右分别是原图、人头标注图、
真实人群密度图和估计人群密度图
</br>
<b>Figure 4</b>：Results based on Paddlepaddle
</p>

