#define id(x, w,  y)     x * w + y



__kernel void add(__global float* a, __global float* b, __global float* c) {
            
    int gl_id_x  =  get_global_id(0);
    int gl_id_y  =  get_global_id(1);
    int width    =  get_global_size(0);

    int buff_idx =  id(gl_id_x, width, gl_id_y);


    c[buff_idx] = a[buff_idx] + b[buff_idx];

}

__kernel void matmul( 
    __global float* a,  __global float* b, __global float* c, int col_a){


            
    int row_idx =  get_global_id(0);
    int col_idx =  get_global_id(1);
    int row_size =  get_global_size(0);


    int res_idx = row_idx * row_size + col_idx;
    
    for (int idx = 0; idx < col_a; idx++)
        c[res_idx] += a[row_idx * row_size + idx] * b[col_idx + row_size*idx];


}

__kernel void sub(__global float* buffer, float scalar) {
    buffer[get_global_id(0)] -= scalar;
}