//Original: https://jott.live/markdown/m1_webgpu_perf
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = N / 4u;

    let cCol = global_id.x * 2u;  
    if (cCol < ND4) {
        var tmp0 = vec4<f32>();
        var tmp1 = vec4<f32>();
        for (var k = 0u; k < K / 4u; k++) {
          let a = A[k];
          let bidx = k * N + cCol;
          tmp0 += vec4<f32>(a.x) * B[bidx]; 
          tmp0 += vec4<f32>(a.y) * B[bidx + (1u * ND4)]; 
          tmp0 += vec4<f32>(a.z) * B[bidx + (2u * ND4)];
          tmp0 += vec4<f32>(a.w) * B[bidx + (3u * ND4)];

          tmp1 += vec4<f32>(a.x) * B[bidx + 1u];
          tmp1 += vec4<f32>(a.y) * B[bidx + (1u * ND4) + 1u];
          tmp1 += vec4<f32>(a.z) * B[bidx + (2u * ND4) + 1u];
          tmp1 += vec4<f32>(a.w) * B[bidx + (3u * ND4) + 1u];

        }
        C[cCol] = tmp0;
        C[cCol + 1u] = tmp1;
    }
}
