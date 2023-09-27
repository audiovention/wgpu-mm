@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = N / 4u;
    let KD4 = K / 4u;
    let cRow = global_id.x;
    let cCol = global_id.y;  
    if (cRow < M && cCol < ND4) {
        var tmp = vec4<f32>(0.0);
        for (var k = 0u; k < KD4; k++) {
          let a = A[cRow * KD4 + k];
          
          let bidx = (k * ND4 * 4u) + cCol;
          let b0 = unpack4x8snorm(B[bidx]) * {{ scale }};
          let b1 = unpack4x8snorm(B[bidx + (1u * ND4)]) * {{ scale }};
          let b2 = unpack4x8snorm(B[bidx + (2u * ND4)]) * {{ scale }};
          let b3 = unpack4x8snorm(B[bidx + (3u * ND4)]) * {{ scale }};

          tmp += vec4<f32>(a.x) * b0;
          tmp += vec4<f32>(a.y) * b1;
          tmp += vec4<f32>(a.z) * b2;
          tmp += vec4<f32>(a.w) * b3;
        }
        C[cRow * ND4 + cCol] = tmp;
    }
}
