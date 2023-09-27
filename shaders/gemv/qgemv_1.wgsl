@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(num_workgroups) num_groups: vec3<u32>,
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = {{ ND4 }}u;
    let KD4 = {{ KD4 }}u;

    let cCol = (group_id.x * {{ workgroup_size_x * workgroup_size_y }}u + local_index) * {{ colPerThread }}u; 
    if (cCol < ND4) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < KD4; k++) {
          let a = A[k];
          let bidx = (k * ND4 * 4u) + cCol;

            tmp += vec4<f32>(a.x) * unpack4x8snorm(B[bidx]) * {{ scale }};
            tmp += vec4<f32>(a.y) * unpack4x8snorm(B[bidx + (1u * ND4)]) * {{ scale }};
            tmp += vec4<f32>(a.z) * unpack4x8snorm(B[bidx + (2u * ND4)]) * {{ scale }};
            tmp += vec4<f32>(a.w) * unpack4x8snorm(B[bidx + (3u * ND4)]) * {{ scale }};
        }
        C[cCol] = tmp;
    }
}


