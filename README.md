# wgpu-mm

How many flops can we squeeze out of wgpu?

This is basically a direct port of Bram Wasti's blog post [here](https://jott.live/markdown/webgpu_safari).
All credits to him.

## GEMM

The M1 8 core GPU can supposedly hit 2.6 TFLOPS of FP32.

A custom metal shader from [Tinygrad](https://github.com/geohot/tinygrad) can
hit 2000 GFLOPS or ~75% of theoretical peak. This shader uses SIMD groups which
WebGPU doesn't support yet - but it's been proposed a few times e.g [here](https://github.com/gpuweb/gpuweb/issues/3950).

The [best shader we have](https://github.com/FL33TW00D/wgpu-mm/tree/master/shaders/gemm/tfjs.wgsl) is an altered version of that by Tensorflow.JS, which reaches ~900GFLOP on my M1.

## GEMV

GEMV is a different problem since it is entirely memory-bound.

We use the formula for bandwidth to be M (GB/s) = M=10-9.(m.n+m+n)*sizeof(scalar type)/T.
For the problem size [1,384] @ [384, 51865] (Whisper logits GEMV), we can calculate the minimum possible runtime to be 1198266.33ns.

The best kernel in here, gemv_2, hits ~1300000ns.

## Read More 

[NVIDIA Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-perf)
