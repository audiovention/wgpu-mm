@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec2<u32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
  let x0 = unpack2x16float(x.x);
  let x1 = unpack2x16float(x.y);
  return vec4<f32>(x0.x, x0.y, x1.x, x1.y);
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = N / 4u;
    let cRow = global_id.x;
    let cCol = global_id.y;  
    if (cRow < M && cCol < N / 4u) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < K / 4u; k++) {
          let a = A[cRow * K / 4u + k];
          let bidx = k * N + cCol;
          let b0 = unpack4x16float(B[bidx]);
          let b1 = unpack4x16float(B[bidx + N/2u]);
          let b2 = unpack4x16float(B[bidx + 2u * N/2u]);
          let b3 = unpack4x16float(B[bidx + 3u * N/2u]);

          tmp += vec4<f32>(a.x) * b0;
          tmp += vec4<f32>(a.y) * b1;
          tmp += vec4<f32>(a.z) * b2;
          tmp += vec4<f32>(a.w) * b3;
        }
        C[cRow * N / 4u + cCol] = tmp; 
    }
}
