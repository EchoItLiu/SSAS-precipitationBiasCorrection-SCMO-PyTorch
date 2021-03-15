from __future__ import absolute_import, division

import torch
import torch.nn as nn
import sys
sys.path.append("/home/yqliu/EC-BiasCorrecting/pre_training/deformConv")
import numpy as np
from deform_conv import th_batch_map_offsets, th_generate_grid

# 注意继承的是Conv2d,先初始化权重后，然后再module.state_dict(torch.load(xx.pt)),不用担心
# 参数不准
class ConvOffset2D(nn.Conv2d):
    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        
        self.filters = filters
        self._grid_param = None
        super(ConvOffset2D, self).__init__(self.filters, self.filters*2, 3, padding=1, bias=False, **kwargs)
        self.weight.data.copy_(self._init_weights(self.weight, init_normal_stddev))

    def forward(self, x):
        """Return the deformed featured map   b0 = b*c"""
        x_shape = x.size()
        offsets = super(ConvOffset2D, self).forward(x)
        # offsets:(b*c, h, w, 2)
        offsets = self._to_bc_h_w_2(offsets, x_shape)
        # x: (b*c, h, w)
        x = self._to_bc_h_w(x, x_shape)
        # outputs: b0 * n_points
        x_offset = th_batch_map_offsets(x, offsets, grid=self._get_grid(self,x))
        #(b0, n_points) → (b, c, h, w)
        x_offset = self._to_b_c_h_w(x_offset, x_shape)
        return x_offset
    
    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(1), x.size(2)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid
    
    # 静态方法可以不传入self方法
    @staticmethod 
    def _init_weights(weights, std):
        fan_out = weights.size(0)
        fan_in = weights.size(1) * weights.size(2) * weights.size(3)
        w = np.random.normal(0.0, std, (fan_out, fan_in))
        return torch.from_numpy(w.reshape(weights.size()))
    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x
    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x
    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x
