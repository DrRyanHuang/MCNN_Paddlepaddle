# -*- coding: utf-8 -*-
"""
@author: RyanHuang
@github: DrRyanHuang


@updateTime: 2020.7.26
@brife: 复现论文MCNN的主干结构
@notice:
    If you have suggestions or find bugs, please be sure to tell me. Thanks!
"""
from paddle.fluid.dygraph import Pool2D, Conv2D, Linear, Conv2DTranspose, BatchNorm
from paddle.fluid.layers import concat, stack
import paddle.fluid as fluid

class ConvBNLayer(fluid.dygraph.Layer):
    '''
    @Brife:
        `ConvBNLayer` 类就是简单的 `Conv + BN` 类
    @Notice:
        `Conv` 无激活函数, `BN` 那儿有激活函数
    '''
    def __init__(self,
                 num_filters,
                 num_channels,
                 filter_size,
                 stride=1,
                 groups=1,              # group参数暂时不用改
                 act='relu',
                 padding=None,
                 name_scope=None,
                 use_bias=False):

        super(ConvBNLayer, self).__init__(name_scope)

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1)//2 if padding is None else padding,
            groups=groups,
            act=None,
            bias_attr=use_bias)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y



class CONVLMS(fluid.dygraph.Layer):

    def __init__(self, channel_list, filter_list, use_bias=False, name_scope=None):
        '''
        @Brife:
            建立MCNN中任意一列的CNN, 用于为后续的MCNN做准备
        @Param:
            channel_list : MCNN中，每一列CNN有4次卷积，该变量用于存放4次卷积的卷积核个数
            filter_list  : MCNN中，每一列CNN有4次卷积，该变量用于存放4次卷积的卷积核size
            use_bias     : 是够在每一层卷积中使用偏置
            name_scope   : 命名空间
        @Return:
            当前列的输出
        @Notice:
            None
        '''
        super(CONVLMS, self).__init__(name_scope)

        self.conv_bn1 = ConvBNLayer(
                num_channels=3, 
                num_filters=channel_list[0], 
                filter_size=filter_list[0], 
                padding=filter_list[0]//2, 
                act='relu',
                name_scope=self.full_name()
                )

        self.conv_bn2 = ConvBNLayer(
                num_channels=channel_list[0], 
                num_filters=channel_list[1], 
                filter_size=filter_list[1], 
                padding=filter_list[1]//2, 
                act='relu', 
                name_scope=self.full_name(),
				)

        self.pool1 = Pool2D(
                pool_size = 2,
                pool_type = 'max',
                pool_stride = 2,
                global_pooling = False,)

        self.conv_bn3 = ConvBNLayer(
                num_channels=channel_list[1], 
                num_filters=channel_list[2], 
                filter_size=filter_list[2], 
                padding=filter_list[2]//2, 
                act='relu',
                name_scope=self.full_name()
				)

        self.pool2 = Pool2D(
                pool_size = 2,
                pool_type = 'max',
                pool_stride = 2,
                global_pooling = False,)

        self.conv_bn4 = ConvBNLayer(
                num_channels=channel_list[2], 
                num_filters=channel_list[3], 
                filter_size=filter_list[3], 
                padding=filter_list[3]//2, 
                act='relu',
                name_scope=self.full_name()
				)

    def forward(self, inputs):

        conv_bn1 = self.conv_bn1(inputs)
        # print(conv_bn1.shape)

        conv_bn2 = self.conv_bn2(conv_bn1)
        # print(conv_bn2.shape)

        pool1 = self.pool1(conv_bn2)
        # print(pool1.shape, 'pool')

        conv_bn3 = self.conv_bn3(pool1)
        # print(conv_bn3.shape)

        pool2 = self.pool2(conv_bn3)
        # print(pool2.shape, 'pool')

        conv_bn4 = self.conv_bn4(pool2)
        # print(conv_bn4.shape)

        return conv_bn4



class MCNN(fluid.dygraph.Layer):

    
    def __init__(self, use_bias=False, name_scope=None):
        
        super(MCNN, self).__init__(name_scope)
        
        channel_list_L = [16, 32, 16, 8]
        filter_list_L = [9, 7, 7, 7]
        self.CNN_L = CONVLMS(channel_list_L, filter_list_L, self.full_name())
        
        channel_list_M = [20, 40, 20, 10]
        filter_list_M = [7, 5, 5, 5]
        self.CNN_M = CONVLMS(channel_list_M, filter_list_M, self.full_name())
        
        channel_list_S = [24, 48, 24, 12]
        filter_list_S = [5, 3, 3, 3]
        self.CNN_S = CONVLMS(channel_list_S, filter_list_S, self.full_name())
        
        self.convall = Conv2D(
                        num_channels=30, 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)
        

        # -----------------------------------------------------------------------------
        # 以上MCNN结构复现完毕
        # -----------------------------------------------------------------------------             
        self.convS = Conv2D(
                        num_channels=channel_list_S[-1], 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)
        
        self.convM = Conv2D(
                        num_channels=channel_list_M[-1], 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)
        
        self.convL = Conv2D(
                        num_channels=channel_list_L[-1], 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)


    def forward(self, inputs, pre_training=None):
        '''
        @Notice:
            `pre_training` 为 `None`时, 不进行预训练
            为 `'L', 'M', 'S'` 时, 进行对应的预训练
        '''
        if pre_training not in [None, 'L', 'M', 'S']:
            raise ValueError("变量`pre_training`的取值范围为`[None, 'L', 'M', 'S']`")

        cnn_L = self.CNN_L(inputs)
        cnn_M = self.CNN_M(inputs)
        cnn_S = self.CNN_S(inputs)

        if pre_training is None:

            convall_pre = concat([cnn_L, cnn_M, cnn_S], axis=1)
            # print(convall_pre.shape)
            convall = self.convall(convall_pre)
            # print(convall.shape)
            return convall

        # ------------ 进行预训练的部分 ------------

        elif pre_training == 'L':
            conv_pre = self.convL(cnn_L)
        elif pre_training == 'M':
            conv_pre = self.convM(cnn_M)
        else:
            conv_pre = self.convS(cnn_S)

        return conv_pre
        
        
        
        
        
        
        
        