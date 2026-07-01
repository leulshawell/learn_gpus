from pyopencl import _cl as cl

import numpy as np
from math import prod

from typing import Tuple, List
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
    
    def __sub__(self, other):
        return self.sub(other)
    
    def __eq__(self, other):
        return self.np == other.np
    
    @property
    def np(self):
        n, _ = cl.enqueue_map_buffer(q, self.buff, flags=cl.map_flags.WRITE, offset=0, shape=self.shape, dtype=np.float32)
        return n


class Matrice(Premitive):
    def __init__(self, buff: cl.Buffer, dtype: Dtype, shape: Tuple[int]):
        self.shape = shape
        self.buff = buff
        self.dtype = dtype
        self.size = prod(shape) * dtype.bytes

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



    def matmul(self, other: "Matrice") -> "Matrice":
        _buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        res, _ = cl.enqueue_map_buffer(q, _buff, flags=cl.map_flags.WRITE, offset=0, shape=(self.shape[0], other.shape[1]), dtype=np.float32)
        Ops.set_args(Ops.matmul, [self.buff, other.buff, _buff, np.int32(self.shape[1])])

        cl.enqueue_nd_range_kernel(q, Ops.matmul, global_work_size=(self.shape[0], other.shape[0]), local_work_size=(1, 1))
        q.finish()

        return Matrice(_buff, self.dtype, (self.shape[0], other.shape[0]))
    
    def add(self, other: "Matrice")->"Matrice":
        buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        Ops.set_args(Ops.add, [self.buff, other.buff, buff])

        cl.enqueue_nd_range_kernel(q, Ops.add, global_work_size=self.shape, local_work_size=(1, 1))
        q.finish()

        return Matrice(buff, self.dtype, self.shape)
    
    
    
    def sub(self, other) -> "Matrice":
        buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=self.size)
        # res, _ = cl.enqueue_map_buffer(q, buff, flags=cl.map_flags.WRITE, offset=0, shape=self.shape, dtype=np.float32)
        Ops.set_args(Ops.sub, [self.buff, other.buff, buff])


        cl.enqueue_nd_range_kernel(q, Ops.sub, global_work_size=self.shape, local_work_size=(1, 1))
        q.finish()

        return Matrice(buff, self.dtype, self.shape)



a = Matrice.rand_int(1, 5, (4, 4))
b = Matrice.rand_int(1, 5, (4, 4))

c = a @ b

d = a + b


print('============================================Tests=============================================')
print("add", "   Passed" if ((a.np + b.np) == (a + b).np).all() else "Failed")
print("sub", "   Passed" if ((a.np - b.np) == (a - b).np).all() else "Failed")
print("matmul", "Passed" if ((a.np @ b.np) == (a @ b).np).all() else "Failed")





