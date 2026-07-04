from pyopencl import _cl as cl

import numpy as np
from math import prod

from enum import IntEnum, auto

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
with open("kernels.template") as f:
    prog_template = f.read()





@dataclass
class Dtype():
    name: str
    bytes: int
    np: np.dtype
    cl: str

class dtypes:
    float32 = Dtype("float32", 4, np.float32, "float")
    int32 = Dtype("int32", 4, np.int32, "int")
    int8 = Dtype("int8", 1, np.int8, "char")



class Ops(IntEnum): 
    ADD = auto(0)
    SUB = auto()
    MUL = auto()
    MATMUL = auto()


class CLRenderer:
    operators = [ "+", "-", "*", "+" ]
    names = {
        Ops.ADD: "elementwise",
        Ops.SUB: "elementwise",
        Ops.MUL: "elementwise",
        Ops.MATMUL: "matmul"
    }

    def render(op: int, dtype: Dtype):
         return CLRenderer.names[op], prog_template.replace("<dtype>", dtype.cl).replace("<operator>", CLRenderer.operators[op])
         
        


class Program:

    prog_cache ={}

    @staticmethod
    def get_kernel(op: int, dtype: Dtype):
        if (key := f"{dtype.name}:{op}") in Program.prog_cache:
            return Program.prog_cache[key]
        kernel, kernel_src = CLRenderer.render(op, dtype)
        prog = cl._Program(context, kernel_src).build(options_bytes=b"")
        new = cl.Kernel(prog, kernel)
        Program.prog_cache[key] = new
        return new
    

    def get_kernel_with_args(kernel: str, dtype: Dtype, args: List[any]):
        kernel = Program.get_kernel(kernel, dtype)
        Program.set_args(kernel, args)
        return kernel


    @staticmethod
    def set_args(op: cl.Kernel, args: List[any]):
        for i, arg in enumerate(args): op.set_arg(i, arg)





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
        n, _ = cl.enqueue_map_buffer(q, self.buff, flags=cl.map_flags.WRITE, offset=self.offset, shape=self.shape, dtype=self.dtype.np)
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
    def rand_int(cls, start: int, end: int,  shape: Tuple[int], dtype: Dtype=dtypes.int8):
        _buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=prod(shape) * dtype.bytes)
        buf, _ = cl.enqueue_map_buffer(q, _buf, flags=cl.map_flags.WRITE, offset=0, shape=shape, dtype=dtype.np) 
        buf[:] = np.random.randint(start, end, shape)
        return Matrice(_buf, dtypes.int32, shape)
    
    @classmethod
    def rand_float(cls, shape: Tuple[int], dtype:Dtype=dtypes.float32):
        _buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=prod(shape) * dtype.bytes)
        buf, _ = cl.enqueue_map_buffer(q, buf, flags=cl.map_flags.WRITE, offset=0, shape=shape, dtype=dtype.np) 
        buf[:] = np.random.rand(*shape, shape).astype(np.float32)
        return Matrice(_buf, dtypes.float32, shape)

    
    def alu(self, other: "Matrice", op: Ops,  dtype: Dtype, shape: Tuple[int], *args,  res_buff: Optional[cl.Buffer]=None):
        buff = res_buff if res_buff else cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        kernel = Program.get_kernel_with_args(op, dtype, [self.buff, other.buff, buff, *args])

        cl.enqueue_nd_range_kernel(q, kernel, global_work_size=shape, local_work_size=(1, 1))
        q.finish()
        return Matrice(buff, self.dtype, shape)


    def matmul(self, other: "Matrice") -> "Matrice":
        assert self.shape[1] == other.shape[0], "Matmul op requires t1.cols == t2.rows"
        return self.alu(other, Ops.MATMUL, self.dtype, (self.shape[1], other.shape[1]), np.int32(self.shape[1]))
    
    def add(self, other: "Matrice")->"Matrice":
        assert self.shape == other.shape, "Add op requires tensor of same dimension"
        return self.alu(other, Ops.ADD, self.dtype, (self.shape[1], other.shape[1]))
        
    
    def sub(self, other: "Matrice") -> "Matrice":
        assert self.shape == other.shape, "Sub op requires tensor of same dimension"
        return self.alu(other, Ops.SUB, self.dtype, (self.shape[1], other.shape[1]))
    
    def mul(self, other: "Matrice") -> "Matrice":
        assert self.shape == other.shape, "mul op requires tensor of same dimension"
        return self.alu(other, Ops.MUL, self.dtype, (self.shape[1], other.shape[1]))
    
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
        res_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.org.size)
        for i, block in enumerate(self.blocks):
            block.mul(other, res_buff.get_sub_region(origin=other.size * i, size=other.size))

        return Matrice(res_buff, self.org.dtype, self.org.shape)

    def sum(self: "BlockedMatrice")->"Matrice":
        pass

    
    def merge() -> "Matrice":
        pass




a = Matrice.rand_int(1, 5, (4, 4))
b = Matrice.rand_int(1, 5, (4, 4))



print('============================================Tests=============================================')
print("add", "   Passed" if ((a.np + b.np) == (a + b).np).all() else "Failed")
print("mul", "   Passed" if ((a.np * b.np) == (a * b).np).all() else "Failed")
print("sub", "   Passed" if ((a.np - b.np) == (a - b).np).all() else "Failed")

print("matmul", "Passed" if ((a.np @ b.np) == (a @ b).np).all() else "Failed")

