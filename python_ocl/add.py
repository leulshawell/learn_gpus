from pyopencl import _cl as cl
import numpy as np


#We do this 
import os


# (Optional) Force the driver to display actual OpenCL compiler output text
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

with open("kernels.cl") as f:
    kernels = f.read()


#Detect and set a default platform/device
plats = cl.get_platforms()
if  not len(plats): print("No OpenCl Device/Driver found"); exit()

default_plat = plats[0]
devices = default_plat.get_devices(cl.device_type.ALL)
default_device = devices[0]




GLOBAL_X, GLOBAL_Y = 8, 8
LOCAL_X, LOCAL_Y,  = 4, 4


DIM = GLOBAL_X * GLOBAL_Y 
LOCAL_WORK_SIZE = LOCAL_X * LOCAL_Y 



print(f"<Device: {default_device.name}>")


context = cl.Context()


a_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=DIM* 4)
b_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=DIM* 4)
c_buff = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.ALLOC_HOST_PTR, size=DIM* 4)



#Create a program (basicaly kernels store with the execution context)
prog = cl._Program(context, kernels).build(options_bytes=b"")


add_kernel = cl.Kernel(prog, "add") #get the add kernel from the program


#Create a command queue (this is for sending commands to the GPU)
q = cl.CommandQueue(context)


#This does NOT copy memory. It just gives Python a direct pointer to the RAM block.
#This creates numpy arrays
a, _ = cl.enqueue_map_buffer(q, a_buff, flags=cl.map_flags.WRITE, offset=0, shape=(DIM, ), dtype=np.float32) 
a.shape = (GLOBAL_X, GLOBAL_Y)


b, _ = cl.enqueue_map_buffer(q, b_buff, flags=cl.map_flags.WRITE, offset=0, shape=(DIM, ), dtype=np.float32)
b.shape = (GLOBAL_X, GLOBAL_Y)

c, _ = cl.enqueue_map_buffer(q, c_buff, flags=cl.map_flags.WRITE, offset=0, shape=(DIM, ), dtype=np.float32)
c.shape = (GLOBAL_X, GLOBAL_Y)


#await all the buffer allocation commands before allocating the buffs and scheduling the kernel
q.finish()



#Populate your data directly on the shared physical RAM
a[:] = np.random.randint(1, 100, (GLOBAL_X, GLOBAL_Y)).astype(np.float32)
b[:] = np.random.randint(1, 100, (GLOBAL_X, GLOBAL_Y)).astype(np.float32)



# #Set the arguments to the add kernel (see <repo root>/kernels.cl)
add_kernel.set_arg(0, a_buff)
add_kernel.set_arg(1, b_buff)
add_kernel.set_arg(2, c_buff)


#queue the kernel execution
cl.enqueue_nd_range_kernel(q, add_kernel, global_work_size=(GLOBAL_X, GLOBAL_Y), local_work_size=(LOCAL_X, LOCAL_Y, ))


q.finish()

print(a)
print("======================================")
print(b)
print("======================================")
print(c)