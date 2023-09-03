@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec2<f32>>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let cRow = global_id.x;
    let cCol = global_id.y;
    if (cRow < M && cCol < N / 2u) {
        var tmp = vec2<f32>(0.0);
        for (var k = 0u; k < K; k++) {
          let a = A[cRow * K + k];
            
          let b = unpack2x16float(B[k * N/2u + cCol]);

          tmp.x += a * b.x;
          tmp.y += a * b.y;
        }
        C[cRow * N / 2u + cCol] = tmp;
    }
}
