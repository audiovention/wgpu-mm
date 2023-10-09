#![allow(non_snake_case)]
use std::{borrow::Cow, fmt::Debug};

use num_traits::{AsPrimitive, Float};
use rand::{
    distributions::{uniform::SampleUniform, Standard, Uniform},
    prelude::Distribution,
};
use wgpu::util::DeviceExt;

use crate::{
    quant::{
        float16_dequantize, rand_quantized_gpu_buffer, sint4_dequantize, sint8_dequantize,
        Quantization,
    },
    sgemv::ABSMAX,
    GPUHandle, Profiler, WorkgroupCount, Workload,
};

fn mm_ref(A: &[f32], B: &[f32], C: &mut [f32], dims: (usize, usize, usize)) {
    let (M, N, K) = dims;
    for m in 0..M {
        for n in 0..N {
            let mut res = 0.;
            for k in 0..K {
                res += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = res;
        }
    }
}

async fn check(
    handle: &GPUHandle,
    pipeline: &wgpu::ComputePipeline,
    wgc: &WorkgroupCount,
    dims: (usize, usize, usize),
    quantized: Quantization,
) {
    let (M, N, K) = dims;

    let (A, A_cpu) = rand_gpu_buffer::<f32>(handle.device(), (M, K), true, false);

    let (B, B_cpu) = match quantized {
        Quantization::None => rand_gpu_buffer::<f32>(handle.device(), (K, N), true, false),
        Quantization::Float16 => {
            let (B, B_cpu) =
                rand_quantized_gpu_buffer(handle.device(), (K, N), true, Quantization::Float16);
            let b_dequant = float16_dequantize(&B_cpu.unwrap(), K, N);
            (B, Some(b_dequant))
        }
        Quantization::SInt8 => {
            let (B, B_cpu) =
                rand_quantized_gpu_buffer(handle.device(), (K, N), true, Quantization::SInt8);
            let b_dequant = sint8_dequantize(&B_cpu.unwrap(), ABSMAX, K, N);
            (B, Some(b_dequant))
        }
        Quantization::SInt4 => {
            let (B, B_cpu) =
                rand_quantized_gpu_buffer(handle.device(), (K, N), true, Quantization::SInt4);
            let b_dequant = sint4_dequantize(&B_cpu.unwrap(), ABSMAX, K, N);
            (B, Some(b_dequant))
        }
    };

    let (C, C_cpu) = empty_buffer::<f32>(handle.device(), (M, N), true);
    let mut C_cpu = C_cpu.unwrap();

    mm_ref(&A_cpu.unwrap(), &B_cpu.unwrap(), &mut C_cpu, dims);

    let bind_group = handle
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: A.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: B.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: C.as_entire_binding(),
                },
            ],
        });

    let gpu_out = mm(handle, pipeline, &bind_group, wgc, 1, &C, None, dims).await;

    let mut mae = 0.0;
    for i in 0..M * N {
        let diff = (gpu_out[i] - C_cpu[i]).abs();
        if diff > mae {
            mae = diff;
        }
    }
    let plen = 32;
    println!(
        "GPU\n{:?}\n...\n{:?}",
        &gpu_out[..plen],
        &gpu_out[M * N - plen..]
    );
    println!(
        "CPU\n{:?}\n...\n{:?}",
        &C_cpu[..plen],
        &C_cpu[M * N - plen..]
    );
    println!("Max Absolute Error: {}", mae);
    let acceptable_mae = match quantized {
        Quantization::None => 1e-5,
        Quantization::Float16 => 1e-3,
        Quantization::SInt8 => 1e-3,
        Quantization::SInt4 => 1e-3,
    };
    if mae > acceptable_mae {
        panic!("MAE too high");
    }
}

pub fn generate_weight_data<F: Float + bytemuck::Pod + AsPrimitive<i32> + Debug>(
    M: usize,
    N: usize,
) -> Vec<F>
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(F::from(-10.0).unwrap()..F::from(10.0).unwrap());
    let mut data = vec![F::zero(); M * N];
    for i in 0..M {
        for j in 0..N {
            data[i * N + j] = dist.sample(&mut rng) / F::from(50).unwrap();
        }
    }

    data
}

fn empty_buffer<F: Float + bytemuck::Pod + AsPrimitive<i32> + Debug>(
    device: &wgpu::Device,
    dims: (usize, usize),
    return_cpu: bool,
) -> (wgpu::Buffer, Option<Vec<F>>)
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let (M, N) = dims;
    let data = vec![F::zero(); M * N];
    let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    if return_cpu {
        (gpu_buffer, Some(data))
    } else {
        (gpu_buffer, None)
    }
}

fn rand_gpu_buffer<F: Float + bytemuck::Pod + AsPrimitive<i32> + Debug>(
    device: &wgpu::Device,
    dims: (usize, usize),
    return_cpu: bool,
    readable: bool,
) -> (wgpu::Buffer, Option<Vec<F>>)
where
    Standard: Distribution<F>,
    F: SampleUniform,
{
    let (M, N) = dims;
    let data = generate_weight_data::<F>(M, N);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: if readable {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::STORAGE
        },
    });
    if return_cpu {
        (buffer, Some(data))
    } else {
        (buffer, None)
    }
}

pub async fn test_harness(
    workload: Workload,
    shader: String,
    dims: (usize, usize, usize),
    quantize_b: Quantization,
) {
    let handle = GPUHandle::new().await.unwrap();
    let (M, N, K) = dims;

    let shader_module = unsafe {
        handle
            .device()
            .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
            })
    };

    let pipeline = handle
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

    check(&handle, &pipeline, workload.count(), (M, N, K), quantize_b).await;

    let (A, _) = rand_gpu_buffer::<f32>(handle.device(), (M, K), false, false);

    let B = match quantize_b {
        Quantization::None => rand_gpu_buffer::<f32>(handle.device(), (K, N), false, false).0,
        Quantization::SInt8 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::SInt8).0
        }
        Quantization::SInt4 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::SInt4).0
        }
        Quantization::Float16 => {
            rand_quantized_gpu_buffer(handle.device(), (K, N), false, Quantization::Float16).0
        }
    };

    let (C, _) = rand_gpu_buffer::<f32>(handle.device(), (M, N), false, true);

    let bind_group_entries = [
        wgpu::BindGroupEntry {
            binding: 0,
            resource: A.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: B.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: C.as_entire_binding(),
        },
    ];

    let bind_group = handle
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bind_group_entries,
        });

    let wgc = workload.count();

    //warmup
    let N_WARMUP = 5;
    mm(
        &handle,
        &pipeline,
        &bind_group,
        wgc,
        N_WARMUP,
        &C,
        None,
        dims,
    )
    .await;

    let N_REPEATS = 10;
    let mut profiler = Profiler::new(handle.clone(), N_REPEATS);
    mm(
        &handle,
        &pipeline,
        &bind_group,
        wgc,
        N_REPEATS as _,
        &C,
        Some(&mut profiler),
        dims,
    )
    .await;
}

async fn mm(
    handle: &GPUHandle,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroup_count: &WorkgroupCount,
    N_REPEATS: usize,
    readback: &wgpu::Buffer,
    mut profiler: Option<&mut Profiler>,
    dims: (usize, usize, usize),
) -> Vec<f32> {
    let mut encoder = handle
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        {
            //WARM
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.set_pipeline(pipeline);
            cpass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
        }
        for _ in 0..N_REPEATS {
            //This needs to be inside the loop for timestamps
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: profiler
                    .as_mut()
                    .map(|prof| prof.create_timestamp_queries(dims.0 * dims.1 * dims.2)),
            });
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.set_pipeline(pipeline);
            cpass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
        }
    }

    if let Some(prof) = profiler.as_mut() {
        prof.resolve(&mut encoder)
    }
    handle.queue().submit(Some(encoder.finish()));
    if let Some(prof) = profiler.as_mut() {
        prof.read_timestamps()
    }
    to_cpu(readback, handle).await
}

async fn to_cpu(buffer: &wgpu::Buffer, handle: &GPUHandle) -> Vec<f32> {
    let buffer_slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);

    wgpu::util::DownloadBuffer::read_buffer(
        handle.device(),
        handle.queue(),
        &buffer_slice,
        move |buffer| {
            tx.send(match buffer {
                Ok(bytes) => bytemuck::cast_slice(&bytes)[..].to_vec(),
                _ => panic!("Error reading buffer"),
            })
            .unwrap();
        },
    );
    handle.device().poll(wgpu::Maintain::Wait);
    rx.recv().unwrap()
}
