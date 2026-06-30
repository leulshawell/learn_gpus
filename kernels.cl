__kernel void fill(__global float* a, float val) {

    int global_id =  get_global_id(0);
    a[global_id] = val;

}


__kernel void add(global float* a, global float* b, global float* c) {
            
    int gl_id_x  =  get_global_id(0);

    int gl_id_y  =  get_global_id(1);

    int width    =  get_global_size(0);

    int buff_idx =  gl_id_x * width + gl_id_y;

    c[buff_idx] = a[buff_idx] + b[buff_idx];

}

__kernel void sub(__global float* buffer, float scalar) {
    buffer[get_global_id(0)] -= scalar;
}