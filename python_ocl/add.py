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



A_ROWS, A_COLS = 40, 40

B_ROWS, B_COLS = 40, 40


LOCAL_X, LOCAL_Y,  = 1, 1





print(f"<Device: {default_device.name}>")


context = cl.Context()


a_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=A_ROWS * A_COLS * 4)
b_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=A_ROWS * A_COLS * 4)





#Create a program (basicaly kernels store with the execution context)
prog = cl._Program(context, kernels).build(options_bytes=b"")


add_kernel = cl.Kernel(prog, "add") #get the add kernel from the program
matmul_kernel = cl.Kernel(prog, "matmul") #get the add kernel from the program


#Create a command queue (this is for sending commands to the GPU)
q = cl.CommandQueue(context)


#This does NOT copy memory. It just gives Python a direct pointer to the RAM block.
#This creates numpy arrays
a, _ = cl.enqueue_map_buffer(q, a_buff, flags=cl.map_flags.WRITE, offset=0, shape=(A_ROWS, A_COLS, ), dtype=np.float32) 
a = a.reshape((A_ROWS, A_COLS), copy=False)


b, _ = cl.enqueue_map_buffer(q, b_buff, flags=cl.map_flags.WRITE, offset=0, shape=(B_ROWS, B_COLS, ), dtype=np.float32)
b = b.reshape((B_ROWS, B_COLS), copy=False)


add_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=A_ROWS * B_COLS * 4)
matmul_buff = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR, size=A_ROWS * B_COLS * 4)


#Send result mem allocate commands
add_result, _ = cl.enqueue_map_buffer(q, add_buff, flags=cl.map_flags.WRITE, offset=0, shape=(A_ROWS, A_COLS), dtype=np.float32)
add_result = add_result.reshape((A_ROWS, A_COLS), copy=False)

matmul_result, _ = cl.enqueue_map_buffer(q, matmul_buff, flags=cl.map_flags.WRITE, offset=0, shape=(A_ROWS, B_COLS), dtype=np.float32)
matmul_result = matmul_result.reshape((A_ROWS, B_COLS), copy=False)



#await all the buffer allocation commands before allocating the buffs and scheduling the kernel
q.finish()



#Populate your data directly on the shared physical RAM
a[:] = np.random.randint(1, 5, (A_ROWS, A_COLS)).astype(np.float32)
b[:] = np.random.randint(1, 5, (B_ROWS, B_COLS)).astype(np.float32)



# #Set the arguments to the add kernel (see <repo root>/kernels.cl)
add_kernel.set_arg(0, a_buff)
add_kernel.set_arg(1, b_buff)
add_kernel.set_arg(2, add_buff)

matmul_kernel.set_arg(0, a_buff)
matmul_kernel.set_arg(1, b_buff)
matmul_kernel.set_arg(2, matmul_buff)
matmul_kernel.set_arg(3, np.int32(A_COLS))


#queue the kernel execution
# cl.enqueue_nd_range_kernel(q, add_kernel, global_work_size=(A_ROWS, A_COLS), local_work_size=(LOCAL_X, 1))
cl.enqueue_nd_range_kernel(q, matmul_kernel, global_work_size=(A_ROWS, B_COLS), local_work_size=(LOCAL_X, LOCAL_Y))


q.finish()

print("==========A & B===================")
print(a)
print(b)
print("===========ADD=======================")
print(add_result)
print("===========MATMUL========================")
print(matmul_result)