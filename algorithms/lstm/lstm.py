from typing import Optional, Tuple

import torch
from torch import nn


class LSTMCell(nn.Module):
    """
    ## Long Short-Term Memory Cell
    """

    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()

        # 上面所示的W_i W_f W_o W_C 四个线性层运算可以用一个线性层进行合并
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        # 是否使用layer normalizations
        # 使用layer normalizations可以获得更好的结果
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # c和h为上一时刻的state，x为当前时刻的输入
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        # 输出的结果代表四个门的各自输出拼接在一起，所以我们在这将其拆开
        ifgo = ifgo.chunk(4, dim=-1)

        # 使用layer normalizations，非必要
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LSTM(nn.Module):
    """
    ## Multilayer LSTM
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        """
        创建有n_layers层LSTMCell 的 LSTM.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # 对每一层创建LSTMCell，第一层要特殊处理
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x 的输出形状为：[n_steps, batch_size, input_size]
        state 包含了传递过来的state即(h, c), h和c各自的形状为[batch_size, hidden_size].
        """
        n_steps, batch_size = x.shape[:2]

        # 第一层的输入需要特殊处理
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        # 创建用来收集每个time step下的结果输出
        out = []
        for t in range(n_steps):
            # 第一个时刻的输入为自己
            inp = x[t]
            # 遍历每个layer
            for layer in range(self.n_layers):
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                # 将当前层的输出当作下一层的输入
                inp = h[layer]
            # 收集最后一个时刻的输出
            out.append(h[-1])

        # 将所有的输出叠加起来
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)


if __name__ == "__main__":
    LSTM = LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = LSTM(input, (h0, c0))
    print(output.shape)
    print(hn.shape)
    print(cn.shape)
