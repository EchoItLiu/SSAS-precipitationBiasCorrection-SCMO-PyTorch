from __future__ import absolute_import, division

import torch
from torch.autograd import Variable

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates


def th_flatten(a):
    return a.contiguous().view(a.nelement())


def th_repeat(a, repeats, axis=0):
    assert len(a.size()) == 1
    # (b0*n_points)
    return th_flatten(torch.transpose(a.repeat(repeats, 1), 0, 1))


def np_repeat_2d(a, repeats):
    assert len(a.shape) == 2
    a = np.expand_dims(a, 0)
    a = np.tile(a, [repeats, 1, 1])
    return a


def _get_vals_by_coords(input, coords, batch_size, n_coords, idx):
    # (b0*n_points, 3)
    indices = torch.stack([
        idx, th_flatten(coords[..., 0]), th_flatten(coords[..., 1])
    ], 1)
    # idx * H * W + offset-x(tblr) * W + offset-y
    inds = indices[:, 0]*input.size(1)*input.size(2)+ indices[:, 1]*input.size(2) + indices[:, 2]
    # 对于b0*H*W大小的feature map，用卷积inds位置标识来映射对应位置像素点值,完成像素重组
    vals = th_flatten(input).index_select(0, inds)
    # 还原新的feature map大小 → b0 * n_points
    vals = vals.view(batch_size, n_coords)
    return vals


def th_batch_map_coordinates(input, coords, order=1):
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)
    # n_points
    n_coords = coords.size(1)
    coords = torch.cat((torch.clamp(coords.narrow(2, 0, 1), 0, input_height - 1), torch.clamp(coords.narrow(2, 1, 1), 0, input_width - 1)), 2)
    assert (coords.size(1) == n_coords)
    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    # (b0, n_points, 1)
    coords_lb = torch.stack([coords_lt[..., 0], coords_rb[..., 1]], 2)
    coords_rt = torch.stack([coords_rb[..., 0], coords_lt[..., 1]], 2)
    idx = th_repeat(torch.arange(0, batch_size), n_coords).long()
    
    idx = Variable(idx, requires_grad=False)
    if input.is_cuda:
        idx = idx.cuda()
    
    # 四个位置(lt rb lb rt)的feature map重组   b0 * n_points
    vals_lt = _get_vals_by_coords(input, coords_lt.detach(), batch_size, n_coords, idx)
    vals_rb = _get_vals_by_coords(input, coords_rb.detach(), batch_size, n_coords, idx)
    vals_lb = _get_vals_by_coords(input, coords_lb.detach(), batch_size, n_coords, idx)
    vals_rt = _get_vals_by_coords(input, coords_rt.detach(), batch_size, n_coords, idx)
    
    # 双线性插值插值 
    coords_offset_lt = coords - coords_lt.type(coords.data.type())
    vals_t = coords_offset_lt[..., 0]*(vals_rt - vals_lt) + vals_lt
    vals_b = coords_offset_lt[..., 0]*(vals_rb - vals_lb) + vals_lb
    ## mapped_vals:  b0 * n_points
    mapped_vals = coords_offset_lt[..., 1]* (vals_b - vals_t) + vals_t
    return mapped_vals


def th_generate_grid(batch_size, input_height, input_width, dtype, cuda):
    grid = np.meshgrid(
        range(input_height), range(input_width), indexing='ij'
    )
    grid = np.stack(grid, axis=-1)
    grid = grid.reshape(-1, 2)
    grid = np_repeat_2d(grid, batch_size)
    grid = torch.from_numpy(grid).type(dtype)    
    if cuda:
        grid = grid.cuda()
        
    return Variable(grid, requires_grad=False)


def th_batch_map_offsets(input, offsets, grid=None, order=1):
    batch_size = input.size(0)
    input_height = input.size(1)
    input_width = input.size(2)
    # b * n_points * 2
    offsets = offsets.view(batch_size, -1, 2)
    if grid is None:
        grid = th_generate_grid(batch_size, input_height, input_width, offsets.data.type(), offsets.data.is_cuda)
    # p0 + pn + △pn相当于在原来的位置上加上了offset 
    coords = offsets + grid    
    ## mapped_vals:  b0 * n_points
    mapped_vals = th_batch_map_coordinates(input, coords)
    return mapped_vals
