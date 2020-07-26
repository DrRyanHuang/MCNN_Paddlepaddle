# -*- coding: utf-8 -*-
"""
@author: RyanHuang
@github: DrRyanHuang


@updateTime: 2020.7.26
@brife: 复现论文MCNN的主干结构
@notice:
    If you have suggestions or find bugs, please be sure to tell me. Thanks!
"""

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

        self.conv1 = Conv2D(
                num_channels=3, 
                num_filters=channel_list[0], 
                filter_size=filter_list[0], 
                stride=1, 
                padding=filter_list[0]//2, 
                act=None, 
                bias_attr=use_bias
        )

        self.batch_norm1 = fluid.BatchNorm(channel_list[0], act='relu')

        self.conv2 = Conv2D(
                num_channels=channel_list[0], 
                num_filters=channel_list[1], 
                filter_size=filter_list[1], 
                stride=1, 
                padding=filter_list[1]//2, 
                act=None, 
                bias_attr=use_bias)
        
        self.batch_norm2 = fluid.BatchNorm(channel_list[1], act='relu')

        self.pool1 = Pool2D(
                pool_size = 2,
                pool_type = 'max',
                pool_stride = 2,
                global_pooling = False,)

        self.conv3 = Conv2D(
                num_channels=channel_list[1], 
                num_filters=channel_list[2], 
                filter_size=filter_list[2], 
                stride=1, 
                padding=filter_list[2]//2, 
                act=None, 
                bias_attr=use_bias)

        self.batch_norm3 = fluid.BatchNorm(channel_list[2], act='relu')

        self.pool2 = Pool2D(
                pool_size = 2,
                pool_type = 'max',
                pool_stride = 2,
                global_pooling = False,)

        self.conv4 = Conv2D(
                num_channels=channel_list[2], 
                num_filters=channel_list[3], 
                filter_size=filter_list[3], 
                stride=1, 
                padding=filter_list[3]//2, 
                act=None, 
                bias_attr=use_bias)
        
        self.batch_norm4 = fluid.BatchNorm(channel_list[3], act='relu')


    def forward(self, inputs):

        conv1 = self.conv1(inputs)
        # print(conv1.shape)
        conv1 = self.batch_norm1(conv1)
        # print(conv1.shape)

        conv2 = self.conv2(conv1)
        # print(conv2.shape)
        conv2 = self.batch_norm2(conv2)
        # print(conv2.shape)

        pool1 = self.pool1(conv2)
        # print(pool1.shape, 'pool')

        conv3 = self.conv3(pool1)
        # print(conv3.shape)
        conv3 = self.batch_norm3(conv3)
        # print(conv3.shape)

        pool2 = self.pool2(conv3)
        # print(pool2.shape, 'pool')

        conv4 = self.conv4(pool2)
        # print(conv4.shape)
        conv4 = self.batch_norm4(conv4)
        # print(conv4.shape)

        return conv4




class MCNN(fluid.dygraph.Layer):

    
    def __init__(self, use_bias=False):
        
        super(MCNN,self).__init__()
        
        channel_list_L = [16, 32, 16, 8]
        filter_list_L = [9, 7, 7, 7]
        self.CNN_L = CONVLMS(channel_list_L, filter_list_L, 'CNN_L')
        
        channel_list_M = [20, 40, 20, 10]
        filter_list_M = [7, 5, 5, 5]
        self.CNN_M = CONVLMS(channel_list_M, filter_list_M, 'CNN_M')
        
        channel_list_S = [24, 48, 24, 12]
        filter_list_S = [5, 3, 3, 3]
        self.CNN_S = CONVLMS(channel_list_S, filter_list_S, 'CNN_S')
        
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
                        num_channels=12, 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)
        
        self.convM = Conv2D(
                        num_channels=10, 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)
        
        self.convL = Conv2D(
                        num_channels=8, 
                        num_filters=1, 
                        filter_size=1,
                        stride=1, 
                        padding=0, 
                        act='relu', 
                        bias_attr=use_bias)


    def forward(self, inputs, ):
    
        cnn_L = self.CNN_L(inputs)
        cnn_M = self.CNN_M(inputs)
        cnn_S = self.CNN_S(inputs)
        
        conv_L = self.convL(cnn_L)
        conv_M = self.convM(cnn_M)
        conv_S = self.convS(cnn_S)
        
        convall_pre = concat([cnn_L, cnn_M, cnn_S], axis=1)
        # print(convall_pre.shape)
        convall = self.convall(convall_pre)
        # print(convall.shape)
        
        return convall, conv_L, conv_M, conv_S