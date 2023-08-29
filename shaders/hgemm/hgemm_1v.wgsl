@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec2<u32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = global_id.x;
    let cCol = global_id.y;  
    if (cRow < M && cCol < N / 4u) {
        var tmp0 = vec4<f32>();
        for (var k = 0u; k < K / 4u; k = k + 1u) {
          let a = A[cRow * K / 4u + k];
          let bidx = k * N + cCol;
          let b0_0 = unpack2x16float(B[bidx].x);
          let b0_1 = unpack2x16float(B[bidx].y);
          let b0 = vec4<f32>(b0_0.x, b0_0.y, b0_1.x, b0_1.y);
          
          let b1_0 = unpack2x16float(B[bidx + (1u * N/4u)].x);
          let b1_1 = unpack2x16float(B[bidx + (1u * N/4u)].y);
          let b1 = vec4<f32>(b1_0.x, b1_0.y, b1_1.x, b1_1.y);

          let b2_0 = unpack2x16float(B[bidx + (2u * N/4u)].x);
          let b2_1 = unpack2x16float(B[bidx + (2u * N/4u)].y);
          let b2 = vec4<f32>(b2_0.x, b2_0.y, b2_1.x, b2_1.y);

          let b3_0 = unpack2x16float(B[bidx + (3u * N/4u)].x);
          let b3_1 = unpack2x16float(B[bidx + (3u * N/4u)].y);
          let b3 = vec4<f32>(b3_0.x, b3_0.y, b3_1.x, b3_1.y);

          tmp0 += vec4<f32>(a.x) * b0;
          tmp0 += vec4<f32>(a.y) * b1;
          tmp0 += vec4<f32>(a.z) * b2;
          tmp0 += vec4<f32>(a.w) * b3;
        }
        C[cRow * N / 4u + cCol] = tmp0; 
    }
}
