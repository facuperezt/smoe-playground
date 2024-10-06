from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import torch


__all__ = [
    'Img2Block',
    'Block2Img',
    'sliding_window',
    'sliding_window_torch',
]


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple, 
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size 
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the 
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an 
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a

    from http://www.johnvinyard.com/blog/?p=268
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every 
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError('ws cannot be larger than a in any dimension. a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)



    import torch

def norm_shape_torch(shape):
    if isinstance(shape, int):
        return (shape,)
    elif isinstance(shape, (tuple, list)):
        return tuple(shape)
    elif isinstance(shape, torch.Tensor):
        return tuple(shape.tolist())
    else:
        raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window_torch(a: torch.Tensor, ws, ss=None, flatten=True):
    if ss is None:
        ss = ws
    ws = norm_shape_torch(ws)
    ss = norm_shape_torch(ss)

    # Convert to torch tensors
    ws = torch.tensor(ws, dtype=torch.int8)
    ss = torch.tensor(ss, dtype=torch.int8)

    # Compute new shape and strides
    newshape = norm_shape_torch((torch.tensor(a.shape) - ws) // ss + 1) + norm_shape_torch(ws)
    newstrides = norm_shape_torch(ss * torch.tensor(a.stride())) + a.stride()

    # Create a strided tensor
    strided = a.as_strided(size=newshape, stride=newstrides)

    if not flatten:
        return strided

    # Flatten the slices
    meat = len(ws) if ws.shape[0] else 0
    firstdim = (torch.prod(torch.tensor(newshape[:-meat]), device=a.device),) if ws.shape[0] else ()
    dim = firstdim + newshape[-meat:]
    # Remove dimensions with size 1
    # dim = [i for i in dim if i != 1]
    return strided.squeeze()

class BlockImgBlock(torch.nn.Module):
    # Variable to keep track of the operations
    _last_unfolded_shape = None

    def __init__(self, block_size: int, img_size: int, n_channels: int = 1, require_grad: bool = False):
        super().__init__()
        self.block_size = block_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.require_grad = require_grad

    def img_to_blocks(self, img_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        device = img_input.device
        return torch.tensor(
                sliding_window(np.asarray(img_input.detach().squeeze().cpu()), 2*[self.block_size], 2*[self.block_size], False),
                requires_grad=self.require_grad,
                dtype=torch.float32,
            ).flatten(0, -3).to(device)

    def blocks_to_img(self, blocked_input: torch.Tensor) -> torch.Tensor:
        if blocked_input.ndim == 4:
            return torch.stack([self.blocks_to_img(x) for x in blocked_input])
        reshape_size = (int(self.img_size/self.block_size), int(self.img_size/self.block_size), self.block_size, self.block_size)
        return blocked_input.reshape(reshape_size).permute(0, 2, 1, 3).reshape(1, self.img_size, self.img_size)

    def visualize_output(self, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
        img = self.blocks_to_img(blocked_output).detach().numpy()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

    def unfold_tensor (self, x, step_c, step_h, step_w):
        kc, kh, kw = step_c, step_h, step_w  # kernel size
        dc, dh, dw = step_c, step_h, step_w  # stride
        
        nc, remainder = np.divmod(x.size(1), kc)
        nc += bool(remainder)
        
        nh, remainder = np.divmod(x.size(2), kh)
        nh += bool(remainder)
        
        nw, remainder = np.divmod(x.size(3), kw)
        nw += bool(remainder)    
        
        pad_c, pad_h, pad_w = nc*kc - x.size(1),  nh*kh - x.size(2), nw*kw - x.size(3)
        x = torch.nn.functional.pad(x, ( 0, pad_h, 0, pad_w, 0, pad_c))
        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = patches.size()
        patches = patches.reshape(unfold_shape[0]*unfold_shape[1]*unfold_shape[2]*unfold_shape[3], unfold_shape[4], unfold_shape[5], unfold_shape[6])
        BlockImgBlock._last_unfolded_shape = torch.tensor(unfold_shape)
        # print(unfold_shape)
        return patches

    def fold_tensor (self, x, shape_x: torch.Tensor = None):
        if shape_x is None:
            shape_x = BlockImgBlock._last_unfolded_shape
        x = x.reshape(shape_x[0], shape_x[1], shape_x[2], shape_x[3], shape_x[4], shape_x[5], shape_x[6])
        x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        #Fold
        output_c = shape_x[1] * shape_x[4]
        output_h = shape_x[2] * shape_x[5]
        output_w = shape_x[3] * shape_x[6]
        x = x.view(-1, output_c, output_h, output_w)
        return x

class Img2Block(BlockImgBlock):
    def forward_old(self, x: torch.Tensor) -> torch.Tensor:
        if 2 <= x.ndim <= 3:
            return self.img_to_blocks(x)
        elif x.ndim == 4:
            return torch.stack([self.img_to_blocks(x[i]) for i in range(x.shape[0])])
        
    def forward_new(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            x = x[None]
        out = self.unfold_tensor(x, self.n_channels, self.block_size, self.block_size)
        return out
    
    def forward(self, x: torch.Tensor, use_old: bool = False) -> torch.Tensor:
        if use_old:
            return self.forward_old(x)
        return self.forward_new(x)

class Block2Img(BlockImgBlock):
    def forward_old(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks_to_img(x)
    
    def forward_new(self, x: torch.Tensor) -> torch.Tensor:
        return self.fold_tensor(x)
    
    def forward(self, x: torch.Tensor, use_old: bool = False) -> torch.Tensor:
        if use_old:
            return self.forward_old(x)
        return self.forward_new(x)