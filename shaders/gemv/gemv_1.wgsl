//Original: https://jott.live/markdown/m1_webgpu_perf
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_index) local_invocation_index: u32,
        @builtin(workgroup_id) group_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = N / 4u;
    let cCol = group_id.x * {{ workgroup_size_x * workgroup_size_y }}u + local_invocation_index;  
    if (cCol < ND4) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < K / 4u; k++) {
          let a = A[k];
          tmp += vec4<f32>(a.x) * B[k * N + cCol]; 
          tmp += vec4<f32>(a.y) * B[k * N + cCol + (1u * ND4)]; 
          tmp += vec4<f32>(a.z) * B[k * N + cCol + (2u * ND4)];
          tmp += vec4<f32>(a.w) * B[k * N + cCol + (3u * ND4)];
        }
        C[cCol] = tmp;
    }
}
