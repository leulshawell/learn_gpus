from pyopencl import _cl as cl

import numpy as np
from math import prod

from typing import Tuple, List, Callable, Optional, Generator
from dataclasses import dataclass




#Detect and set a default platform/device
plats = cl.get_platforms()
if  not len(plats): print("No OpenCl Device/Driver found"); exit()

default_plat = plats[0]
devices = default_plat.get_devices(cl.device_type.ALL)
default_device = devices[0]


print(f"<Device: {default_device.name}>")



context = cl.Context()

#Create a command queue (this is for sending commands to the GPU)
q = cl.CommandQueue(context)


#read kernels soure code 
with open("kernels.cl") as f:
    kernel_src = f.read()



class Ops:
    _progs  = cl._Program(context, kernel_src).build(options_bytes=b"")

    matmul = cl.Kernel(_progs, "matmul")
    add = cl.Kernel(_progs, "add")
    mul = cl.Kernel(_progs, "mul")
    sub = cl.Kernel(_progs, "sub")

    @staticmethod
    def set_args(op: cl.Kernel, args: List[any]):
        for i, arg in enumerate(args): op.set_arg(i, arg)





@dataclass
class Dtype:
    name: str
    bytes: int



float32 = Dtype("float32", 4)
int32 = Dtype("int32", 4)



#this is just for code readability
class Premitive:
    def __matmul__(self, other):
        return self.matmul(other)
    
    def __add__(self, other):
        return self.add(other)
    
    def __mul__(self, other):
        return self.mul(other)
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __eq__(self, other):
        return self.np == other.np
    
    @property
    def np(self):
        n, _ = cl.enqueue_map_buffer(q, self.buff, flags=cl.map_flags.WRITE, offset=0, shape=self.shape, dtype=np.float32)
        return n



class Matrice(Premitive):
    def __init__(self, buff: cl.Buffer, dtype: Dtype, shape: Tuple[int], offset=0, stride=None):
        assert len(shape) < 3, "Only support 2D matrices"

        if len(shape) == 1: shape = (1, ) + shape
        self.offset = offset
        self.size = prod(shape) * dtype.bytes

        self.buff = buff if not offset else buff.get_sub_region(origin=offset, size=self.size)
        self.dtype = dtype
        self.shape = shape
        self.stride = stride if stride else tuple(reversed(tuple(prod(tuple(reversed(self.shape))[:dim]) for dim in range(len(shape)))))

    @classmethod
    def rand_int(cls, start: int, end: int,  shape: Tuple[int]):
        _buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=prod(shape) * 4)
        buf, _ = cl.enqueue_map_buffer(q, _buf, flags=cl.map_flags.WRITE, offset=0, shape=shape, dtype=np.float32) 
        buf[:] = np.random.randint(start, end, shape).astype(np.float32)
        return Matrice(_buf, int32, shape)
    
    @classmethod
    def rand_float(cls, shape: Tuple[int]):
        _buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=prod(shape) * 4)
        buf, _ = cl.enqueue_map_buffer(q, buf, flags=cl.map_flags.WRITE, offset=0, shape=shape, dtype=np.float32) 
        buf[:] = np.random.rand(*shape, shape).astype(np.float32)
        return Matrice(_buf, float32, shape)



    def matmul(self, other: "Matrice", res_buff: Optional[cl.Buffer]=None) -> "Matrice":
        assert self.shape[1] == other.shape[0], "Matmul op requires t1.cols == t2.rows"
        _buff = res_buff if res_buff else cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        Ops.set_args(Ops.matmul, [self.buff, other.buff, _buff, np.int32(self.shape[1])])

        cl.enqueue_nd_range_kernel(q, Ops.matmul, global_work_size=(self.shape[0], other.shape[0]), local_work_size=(1, 1))
        q.finish()

        return Matrice(_buff, self.dtype, (self.shape[0], other.shape[0]))
    
    def add(self, other: "Matrice", res_buff: Optional[cl.Buffer]=None)->"Matrice":
        assert self.shape == other.shape, "Add op requires tensor of same dimension"
        buff = res_buff if res_buff else cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        Ops.set_args(Ops.add, [self.buff, other.buff, buff])

        cl.enqueue_nd_range_kernel(q, Ops.add, global_work_size=self.shape, local_work_size=(1, 1))
        q.finish()

        return Matrice(buff, self.dtype, self.shape)
    
    
    
    def sub(self, other: "Matrice", res_buff: Optional[cl.Buffer]=None) -> "Matrice":
        assert self.shape == other.shape, "Sub op requires tensor of same dimension"
        buff = res_buff if res_buff else cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        Ops.set_args(Ops.sub, [self.buff, other.buff, buff])

        cl.enqueue_nd_range_kernel(q, Ops.sub, global_work_size=self.shape, local_work_size=(1, 1))
        q.finish()

        return Matrice(buff, self.dtype, self.shape)
    
    def mul(self, other: "Matrice", res_buff: Optional[cl.Buffer]=None) -> "Matrice":
        assert self.shape == other.shape, "mul op requires tensor of same dimension"
        buff = res_buff if res_buff else cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        Ops.set_args(Ops.mul, [self.buff, other.buff, buff])
        
        cl.enqueue_nd_range_kernel(q, Ops.mul, global_work_size=self.shape, local_work_size=(1, 1))
        q.finish()

        return Matrice(buff, self.dtype, self.shape)
    
    def conv(self, kernel: "Matrice"):
        #                                                                       shape         offset                                stride
        blocked = BlockedMatrice(self, self.size//kernel.size, lambda blk_num: (kernel.shape, self.offset + kernel.size * blk_num, self.stride))
        res = blocked.mul(kernel)
        return res


    def __hash__(self):
        return id(self)
    


class BlockedMatrice:
    """This is a matrice of matrices used to break a bigger matrice in to multiple smaller matrices for ops like conv"""
    def __init__(self, org: "Matrice",  n: int, breaker: Callable[..., Tuple[int, int, int]]):
        self.org = org
        self.block_count = n
        self.blocks: Generator[Matrice] = (Matrice(org.buff, org.dtype, *breaker(i)) for i in range(n))

    def matmul(self, other): 
        return
    
    def mul(self: "BlockedMatrice", other: "Matrice"):
        res_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=other.size * self.block_count)
        for i, block in enumerate(self.blocks):
            block.mul(other, res_buff.get_sub_region(origin=other.size * i, size=other.size))

        return Matrice(res_buff, self.org.dtype, self.org.shape)

    def sum(self: "BlockedMatrice")->"Matrice":
        pass

    
    def merge() -> "Matrice":
        pass





    





import time


a = Matrice.rand_int(1, 5, (4, 4))
b = Matrice.rand_int(1, 5, (4, 4))


d = a.conv(b)

print(a.np)
print("==================================================================")
print(b.np)
print("==================================================================")
print(d.np)



print('============================================Tests=============================================')
print("add", "   Passed" if ((a.np + b.np) == (a + b).np).all() else "Failed")
print("mul", "   Passed" if ((a.np * b.np) == (a * b).np).all() else "Failed")
print("sub", "   Passed" if ((a.np - b.np) == (a - b).np).all() else "Failed")

print("matmul", "Passed" if ((a.np @ b.np) == (a @ b).np).all() else "Failed")

