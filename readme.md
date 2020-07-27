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
<a href=https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf>MCNN</a>是CVPR2016年的一篇论文。作者提出了一种简单而有效的<b>多列卷积神经网络架构(Multi-column Convolutional Neural Network, MCNN)</b>，通过使用大小不同的卷积核去适应人/头部大小的变化，以将图片映射为人群密度图。作者在文章中使用了<b>几何自适应高斯核</b>去计算人群密度图(Ground Truth)，同时收集并标记了一个大型的新数据集(ShanghaiTech数据集)，其中包括1198幅图像，数据集可以在AI Studio上下载到: https://aistudio.baidu.com/aistudio/datasetdetail/10675 .
</p>

<br/>
<br/>

<p>
MCNN受MDNNs的启发，由三列并行的CNN组成，每列CNN卷积核大小不同。为了简化，所有列使用相同的网络结构(即conv-pooling-conv-pooling)。每次池化都会使用2*2的Max Pooling，而激活函数全部选择Relu。堆叠三列CNN的输出特征图，并使用1*1的卷积核将其映射为密度图。MCNN的整体架构图如<b>Figure 1</b>所示：
</p>
<img src="https://github.com/DrRyanHuang/MCNN_Paddlepaddle/blob/master/src/archit.png"  alt="archit" width="1000" height="500"/>

<p align="center"><b>图 1</b>：用于人群密度图估计的多列卷积神经网络(MCNN)的结构</p>
<p align="center"><b>Figure 1</b>：The structure of the proposed multi-column convolutional neural network for crowd density map estimation.</p>

