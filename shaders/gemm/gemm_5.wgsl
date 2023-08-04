//Kernel 5: 2D Blocktiling for Calculating Multiple Results per Thread
//https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh
@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

var<workgroup> As: array<f32, {{ BM * BK }}u>;
var<workgroup> Bs: array<f32, {{ BK * BN }}u>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = group_id.y; 
    let cCol = group_id.x;

    let resultsPerBlock = {{ BM * BN }}u;
    let threadsPerBlock = resultsPerBlock / {{ TM * TN }}u;

    // BN/TN are the number of threads to span a column
    let threadCol = local_id.x % {{ BN / TN }}u;
    let threadRow = local_id.x / {{ BN / TN }}u;

    var aIdx = cRow * {{ BM }}u * K;                    
    var bIdx = cCol * {{ BN }}u;                        
    var cIdx = cRow * {{ BM }}u * N + cCol * {{ BN }}u; 

    let tileColA = local_id.x % {{ BK }}u; 
    let tileRowA = local_id.x / {{ BK }}u;
    let strideA = threadsPerBlock / {{ BK }}u;

    let tileColB = local_id.x % {{ BN }}u; 
    let tileRowB = local_id.x / {{ BN }}u;
    let strideB = threadsPerBlock / {{ BN }}u;

    var threadResults = array<f32, {{ TM * TN }}u>();

    var regM = array<f32, {{ TM }}u>();
    var regN = array<f32, {{ TN }}u>();

    for (var bkIdx = 0u; bkIdx < K; bkIdx += {{ BK }}u) {
        //Each thread loads multiple elements from A and B
        for (var loadOffset = 0u; loadOffset < {{ BM }}u; loadOffset += strideA) {
            As[(tileRowA + loadOffset) * {{ BK }}u + tileColA] = A[aIdx + (tileRowA + loadOffset) * K + tileColA];
        }
        for (var loadOffset = 0u; loadOffset < {{ BK }}u; loadOffset += strideB) {
            Bs[(tileRowB + loadOffset) * {{ BN }}u + tileColB] = B[bIdx + (tileRowB + loadOffset) * N + tileColB];
        }
        workgroupBarrier();

        aIdx += {{ BK }}u;
        bIdx += {{ BK }}u * N;

        for (var dotIdx = 0u; dotIdx < {{ BK }}u; dotIdx++) {
            for (var i = 0u; i < {{ TM }}u; i++) {
                regM[i] = As[(threadRow * {{ TM }}u + i) * {{ BK }}u + dotIdx];
            }
            for (var i = 0u; i < {{ TN }}u; i++) {
                regN[i] = Bs[dotIdx * {{ BN }}u + threadCol * {{ TN }}u + i];
            }
            for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
                for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN++) {
                    threadResults[resIdxM * {{ TN }}u + resIdxN] = fma(regM[resIdxM], regN[resIdxN], threadResults[resIdxM * {{ TN }}u + resIdxN]);
                }
            }
        }
        workgroupBarrier();
    }
    for (var resIdxM = 0u; resIdxM < {{ TM }}u; resIdxM++) {
        for (var resIdxN = 0u; resIdxN < {{ TN }}u; resIdxN++) {
            C[cIdx + (threadRow * {{ TM }}u + resIdxM) * N + threadCol * {{ TN }}u + resIdxN] = threadResults[resIdxM * {{ TN }}u + resIdxN];
        }
    }
}
