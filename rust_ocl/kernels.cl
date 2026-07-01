#define idx(row_idx, row_size,  col_idx)     row_idx * row_size + col_idx


//this assumes worker size = total size of contiguous matrice
__kernel void add(__global float* a, __global float* b, __global float* c) {
            
    int gl_id_x  =  get_global_id(0);
    int gl_id_y  =  get_global_id(1);
    int width    =  get_global_size(0);

    int buff_idx =  idx(gl_id_x, width, gl_id_y);

    c[buff_idx] = a[buff_idx] + b[buff_idx];

}



//this assuumes that you launch workers of the size as your result matrice
__kernel void matmul(__global float* a,  __global float* b, __global float* c, int common_dim){             

    int a_row_idx  =  get_global_id(0);
    int b_col_idx =  get_global_id(1);

    int a_rows =  get_global_size(0);
    int b_col =  get_global_size(1);

    int res_idx = idx(a_row_idx, b_col, b_col_idx);

    int sum = 0; 
    for (int i = 0;   i < common_dim; i++){
        sum += a[idx(a_row_idx, common_dim, i)] * b[idx(i, b_col, b_col_idx)];
    }

    c[res_idx] = sum;

}




//kerne for convolution
__kernel void conv1d(){
    
}

__kernel void sub(__global float* buffer, float scalar) {
    buffer[get_global_id(0)] -= scalar;
}

