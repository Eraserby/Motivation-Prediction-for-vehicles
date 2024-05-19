# import torch
# import torch.nn as nn

# class Seq2SeqModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2SeqModel, self).__init__()
#         self.encoder = nn.Linear(input_size, hidden_size)
#         self.decoder = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input_seq):
#         # Encoder
#         encoder_output = self.encoder(input_seq)

#         # Decoder
#         decoder_output = self.decoder(encoder_output)
#         output_prob = self.softmax(decoder_output)

#         return output_prob

# # 示例数据
# input_data = torch.tensor([[1.0, 2.0],
#                            [3.0, 4.0],
#                            [5.0, 6.0],
#                            [100,500]])  # 输入数据为两列 n 行

# # 初始化模型
# input_size = 2  # 输入数据列数
# hidden_size = 64  # 隐藏层大小
# output_size = 3  # 输出数据行数
# model = Seq2SeqModel(input_size, hidden_size, output_size)

# # 模型前向传播
# output_prob = model(input_data)

# print("输出概率分布：")
# print(output_prob)

import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_seq):
        # Encoder
        encoder_output = self.encoder(input_seq)

        # Decoder
        decoder_output = self.decoder(encoder_output)

        return decoder_output

# 示例数据
input_data = torch.tensor([[1.0, 2.0],
                           [3.0, 4.0],
                           [5.0, 6.0],
                           [2,3],
                           [55,-123]])  # 输入数据为不确定行数，两列

# 初始化模型
input_size = 2  # 输入数据列数
hidden_size = 64  # 隐藏层大小
output_size = 3  # 输出数据列数，行数为一行
model = Seq2SeqModel(input_size, hidden_size, output_size)

# 模型前向传播
output = model(input_data)

print("输出数据：")
print(output)