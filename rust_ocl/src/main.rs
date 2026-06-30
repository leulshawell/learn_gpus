use ocl::{Buffer, MemFlags, OclPrm, ProQue};
use std::fs;





//number of threads to be launched
const  DIM: usize = 512;

//num of threads per group
const LOCAL_WORK_SIZE: usize  = 16; 



fn create_buffer<S: OclPrm>(proq: &ProQue, size: usize, flags: MemFlags) -> ocl::Result<Buffer<S>> {
    Buffer::builder()
        .queue(proq.queue().clone())
        .flags(flags)
        .len(size)
        .build()
}

// int group_size = get_local_size(0);
// int local_id  =  get_local_id(0);
// int group_id  =  get_group_id(0);

fn trivial()->ocl::Result<()>{

    let src = fs::read_to_string("./kernels.cl")?;
//     let src = r#"
//    __kernel void add(__global float* a, __global float* b, __global float* c) {
                
//         int global_id =  get_global_id(0);
//         int local_id  =  get_local_id(0);
//         int group_id  =  get_group_id(0);
//         int group_size = get_local_size(0);

//         for (int i = 0; i < 10000; i ++){
//             c[global_id] = group_size * group_id + local_id;
//         }
//     }

//     __kernel void sub(__global float* buffer, float scalar) {
//         buffer[get_global_id(0)] -= scalar;
//     }
//     "#;

    //create a program queue builder (this just takes our kernels src and queues them)
    let proque = ProQue::builder()
        .src(src)
        .dims(DIM)
        .build()?;


        
    //where the result is going (on th e GPU)
    let a = create_buffer::<f32>(&proque, DIM, MemFlags::new().read_write())?;
    let b = create_buffer::<f32>(&proque, DIM, MemFlags::new().read_write())?;
    let c = create_buffer::<f32>(&proque, DIM, MemFlags::new().read_write())?;

    

    // let kernel_fill = proque.kernel_builder("fill")
    //     .global_work_size(DIM)
    //     .local_work_size(LOCAL_WORK_SIZE)
    //     .build()?;
        
    
    // let fill_a = kernel_fill (&a, 10.0);
        

    // let fill_b = kernel_fill.set_arg(&b, 10.0);
        
    
    // unsafe { 
    //     fill_a.enq()?;
    //     fill_b.enq()?;
    // }


    let kernel = proque.kernel_builder("add")
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .global_work_size((DIM, 1))
        .local_work_size((LOCAL_WORK_SIZE, 1))
        .build()?;

    
    unsafe { kernel.enq()?; }
    let _ = proque.finish();

    let mut vec = vec![0.0f32; c.len()];

    println!("{:>3}", vec.len());

    //copy to the CPU or This rust programs memory
    c.read(&mut vec).enq()?;

    let mut i = 0;


    for &value in &vec {
        if i % LOCAL_WORK_SIZE == 0  {
            print!("\nGROUP {:>3}\t", i / LOCAL_WORK_SIZE);
        }
        i += 1;
        print!("{:>5}", value)
    }

    println!();
    Ok(())
}



fn main(){

    trivial().expect("REASON")
}