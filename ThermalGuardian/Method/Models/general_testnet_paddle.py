# 用于非DLJSFuzzer方法，cell不重复，不添加任何操作
import copy
import random

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GeneralPaddleNet(nn.Layer):
    def __init__(self,in_channel,final_module,channels):
        super(GeneralPaddleNet, self).__init__()
        self.in_channel = in_channel
        self.final_module = final_module
        self.channels = channels
    def forward(self,x):
        # 各节点的张量。
        tensors = []
        # 判断某张量是否有初始值。
        tensors_isnull = [True] * len(self.channels)
        tensors.append(x)
        tensors_isnull[0] = False
        for i in range(len(self.channels) - 1):
            # 随意赋一个同类型的初始值
            tensors.append(True)
        final_point = 0
        for eachOperation in self.final_module:
            fromIndex = eachOperation.fromIndex
            final_point = eachOperation.toIndex
            input = tensors[fromIndex]
            toIndex = eachOperation.toIndex
            # 本NAS规定所有操作出通道数与入通道数相同。
            operator_in_channel = self.channels[fromIndex]*self.in_channel
            operator = eachOperation.operator

            # if operator != -1 and operator != 0:
            #     operator = -1
            # indentity
            if operator == -1:
                # print("paddlepaddle执行了了操作-1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = tensors[fromIndex].clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # concat用于数组之间的操作，因此需要先进行类型转换。
                    temp = paddle.concat([tensors[toIndex], input], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 1 × 1 convolution of C channels
            elif operator == 1:
                # print("paddlepaddle执行了操作1 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是1*1卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是1*1卷积的代码。
                    filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=0)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 depthwise convolution
            elif operator == 2:
                # print("paddlepaddle执行了操作2", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    # 注：paddlepaddle中dw卷积加pw卷积是普通卷积操作加groups来调节的。
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = paddle.nn.ZeroPad2D(padding=1)
                    input = pad(input)
                    thisresult = F.conv2d(input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    # thisresult = F.conv2d(input=input,weight=filter,stride=1,padding=[1,1],groups=operator_in_channel)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    thisresult = F.conv2d(input, weight=filter, stride=1, padding=0,
                                                            groups=operator_in_channel)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 3:
                # print("paddlepaddle执行了操作3", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # filter参数顺序:OutChannel、InChannel/groups、H、W
                    depthwise_filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    pointwise_filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1], dtype=paddle.float32)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = paddle.nn.functional.conv2d(input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = paddle.nn.functional.conv2d(depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = pointwise_temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    depthwise_filter = paddle.ones([operator_in_channel, 1, 3, 3], dtype=paddle.float32)
                    pointwise_filter = paddle.ones([self.in_channel, operator_in_channel, 1, 1], dtype=paddle.float32)
                    pad = paddle.nn.ZeroPad2D(padding=(1, 1, 1, 1))
                    input = pad(input)
                    depthwise_temp = paddle.nn.functional.conv2d(input, weight=depthwise_filter, stride=1,
                                                                padding=0, groups=operator_in_channel).clone().detach()
                    pointwise_temp = paddle.nn.functional.conv2d(depthwise_temp, weight=pointwise_filter, stride=1,
                                                                padding=0).clone().detach()
                    tensors[toIndex] = paddle.concat([tensors[toIndex], pointwise_temp], 1).clone().detach()
            elif operator == 4:
                # print("paddlepaddle执行了了操作4 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.MaxPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=False)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 5:
                # print("paddlepaddle执行了了操作5 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 注意：一定要有count_include_pad=False,不计算补的0，和tensorflow保持一致。
                    result = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=True)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.AvgPool2D(kernel_size=3, stride=1, padding=1, ceil_mode=True)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # 3 × 3 convolution of C channels
            elif operator == 6:
                # print("paddlepaddle执行了操作6 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([self.in_channel, operator_in_channel, 3, 3])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = paddle.ones([self.in_channel, operator_in_channel, 3, 3])
                    thisresult = F.conv2d(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #3*3 conv2D_transpose of C channels
            #注：参数代表的是正向卷积的过程，但我要做的是反卷积。
            elif operator == 7:
                # print("paddlepaddle执行了操作7 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 这是3*3卷积的代码
                    # filter参数顺序:OutChannel、InChannel、H、W
                    filter = paddle.ones([operator_in_channel, self.in_channel, 3, 3])
                    thisresult = F.conv2d_transpose(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = thisresult.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 这是3*3卷积的代码。
                    filter = paddle.ones([operator_in_channel, self.in_channel, 3, 3])
                    thisresult = F.conv2d_transpose(input, weight=filter, stride=[1, 1], padding=1)
                    tensors[toIndex] = paddle.concat([tensors[toIndex], thisresult], 1).clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #relu
            elif operator == 8:
                # print("paddlepaddle执行了了操作8 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = F.relu(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = F.relu(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #sigmoid
            elif operator == 9:
                # print("paddlepaddle执行了了操作9 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.functional.sigmoid(input)
                    tensors[toIndex] = result
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.functional.sigmoid(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #tanh
            elif operator == 10:
                # print("paddlepaddle执行了了操作10 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.functional.tanh(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.functional.tanh(input)
                    temp = paddle.concat((tensors[toIndex], result), 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #leakyrelu
            elif operator == 11:
                # print("paddlepaddle执行了了操作11 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.LeakyReLU(negative_slope=0.2)(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.LeakyReLU(negative_slope=0.2)(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #prelu。
            elif operator == 12:
                # print("paddlepaddle执行了了操作12 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    weight = paddle.ones([operator_in_channel])*0.2
                    result = paddle.nn.functional.prelu(input,weight)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    weight = paddle.ones([operator_in_channel])*0.2
                    result = paddle.nn.functional.prelu(input,weight)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            #ELU
            elif operator == 13:
                # print("paddlepaddle执行了了操作13 ", eachOperation.fromIndex, " ", eachOperation.toIndex)
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    result = paddle.nn.ELU()(input)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    result = paddle.nn.ELU()(input)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            elif operator == 14:  # matmul算子

                # 创建全1的辅助张量（确保计算前后形状不变）
                # 根据输入张量形状创建匹配的单位矩阵[1,3](@ref)
                last_dim = input.shape[-1]
                identity_matrix = paddle.ones(shape=[last_dim, last_dim], dtype=input.dtype)

                # 执行矩阵乘法[1,2](@ref)
                result = paddle.matmul(input, identity_matrix)

                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    tensors[toIndex] = result.clone().detach()
                    # print(f"paddlepaddle执行了操作13（matmul）{eachOperation.fromIndex}->{eachOperation.toIndex}（新建）")
                else:
                    # 沿第1轴拼接结果（保持与ELU算子相同的拼接逻辑）
                    temp = paddle.concat([tensors[toIndex], result], axis=1)
                    tensors[toIndex] = temp.clone().detach()
                    # print(f"paddlepaddle执行了操作13（matmul）{eachOperation.fromIndex}->{eachOperation.toIndex}（拼接）")
            # RNN
            elif operator == 15:
                # 随机确定单向还是多项
                ran = random.random()
                if ran > 0.5:
                    is_reverse = True
                else:
                    is_reverse = False
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 创建与输入形状匹配的RNN单元(RNN不支持动态shape，因此只能硬编码)
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    cell = paddle.nn.SimpleRNNCell(320, 320)  # 隐藏层大小=输入大小
                    rnn_layer = paddle.nn.RNN(cell, is_reverse=is_reverse, time_major=False)  # 设置time_major=False保持形状一致
                    result = rnn_layer(input)[0]  # 只取输出序列
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 创建与输入形状匹配的RNN单元(RNN不支持动态shape，因此只能硬编码input)
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    cell = paddle.nn.SimpleRNNCell(320, 320)  # 隐藏层大小=输入大小
                    rnn_layer = paddle.nn.RNN(cell, is_reverse=is_reverse, time_major=False)  # 设置time_major=False保持形状一致
                    result = rnn_layer(input)[0]  # 只取输出序列
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # LSTM
            elif operator == 16:
                # 随机确定层数和单向多向
                ran = random.random()
                if ran > 0.5:
                    num_layers = 1
                else:
                    num_layers = 2
                ran = random.random()
                if ran > 0.5:
                    direction = "bidirectional"
                else:
                    direction = "forward"

                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 动态创建与输入形状匹配的LSTM
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    # 创建LSTM层（隐藏层大小=输入维度）
                    lstm_layer = paddle.nn.LSTM(
                        input_size=input_size,
                        hidden_size=input_size,  # 关键：设置隐藏层大小=输入维度
                        num_layers=num_layers,  # 指定LSTM层数
                        direction=direction, # 指定单向or多向
                        time_major=False  # 输入格式为[batch_size, seq_len, input_size]
                    )
                    # 增加batch维度（LSTM要求3D输入）
                    input_with_batch = input.unsqueeze(0)
                    # 执行LSTM计算
                    result, _ = lstm_layer(input_with_batch)
                    # 移除batch维度
                    result = result.squeeze(0)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 动态创建与输入形状匹配的LSTM
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    # 创建LSTM层（隐藏层大小=输入维度）
                    lstm_layer = paddle.nn.LSTM(
                        input_size=input_size,
                        hidden_size=input_size,  # 关键：设置隐藏层大小=输入维度
                        num_layers=num_layers,  # 指定LSTM层数
                        direction=direction, # 指定单向or多向
                        time_major=False  # 输入格式为[batch_size, seq_len, input_size]
                    )
                    # 增加batch维度（LSTM要求3D输入）
                    input_with_batch = input.unsqueeze(0)
                    # 执行LSTM计算
                    result, _ = lstm_layer(input_with_batch)
                    # 移除batch维度
                    result = result.squeeze(0)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
            # GRU
            elif operator == 17:
                # 随机确定层数和单向多向
                ran = random.random()
                if ran > 0.5:
                    num_layers = 1
                else:
                    num_layers = 2
                ran = random.random()
                if ran > 0.5:
                    direction = "bidirectional"
                else:
                    direction = "forward"
                if tensors_isnull[toIndex] == True:
                    tensors_isnull[toIndex] = False
                    # 动态创建与输入形状匹配的GRU
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    # 创建GRU层（隐藏层大小=输入维度）
                    gru_layer = paddle.nn.GRU(
                        input_size=input_size,
                        hidden_size=input_size,  # 关键：设置隐藏层大小=输入维度
                        num_layers=num_layers,  # 指定GRU层数
                        direction=direction, # 指定单向or多向
                        time_major=False  # 输入格式为[batch_size, seq_len, input_size]
                    )
                    # 增加batch维度（GRU要求3D输入）
                    input_with_batch = input.unsqueeze(0)
                    # 执行GRU计算
                    result, _ = gru_layer(input_with_batch)
                    # 移除batch维度
                    result = result.squeeze(0)
                    tensors[toIndex] = result.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
                else:
                    # 动态创建与输入形状匹配的GRU
                    input_size = input.shape[-1]  # 获取输入张量最后一维大小
                    # 创建GRU层（隐藏层大小=输入维度）
                    gru_layer = paddle.nn.GRU(
                        input_size=input_size,
                        hidden_size=input_size,  # 关键：设置隐藏层大小=输入维度
                        num_layers=num_layers,  # 指定GRU层数
                        direction=direction, # 指定单向or多向
                        time_major=False  # 输入格式为[batch_size, seq_len, input_size]
                    )
                    # 增加batch维度（GRU要求3D输入）
                    input_with_batch = input.unsqueeze(0)
                    # 执行GRU计算
                    result, _ = gru_layer(input_with_batch)
                    # 移除batch维度
                    result = result.squeeze(0)
                    temp = paddle.concat([tensors[toIndex], result], 1)
                    tensors[toIndex] = temp.clone().detach()
                    # #log
                    # print("tensor"+str(toIndex)+":")
                    # print(tensors[toIndex])
        return tensors[final_point]