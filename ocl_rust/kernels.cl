__kernel void fill(__global float* a, float val) {

    int global_id =  get_global_id(0);
    a[global_id] = val;

}


__kernel void add(__global float* a, __global float* b, __global float* c) {
            
    int global_id =  get_global_id(0);
    int local_id  =  get_local_id(0);
    int group_id  =  get_group_id(0);
    int group_size = get_local_size(0);

    c[global_id] = a[global_id] + b[global_id] ;
}

__kernel void sub(__global float* buffer, float scalar) {
    buffer[get_global_id(0)] -= scalar;
}