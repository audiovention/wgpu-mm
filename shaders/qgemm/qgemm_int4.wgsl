@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<u32>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

fn unpack8x4snorm(value: u32, absmax: f32) -> array<vec4<f32>, 2> {
    let bingo = i32(value);
    let x = f32((bingo << 28u) >> 28u);
    let y = f32((bingo << 24u) >> 28u);
    let z = f32((bingo << 20u) >> 28u);
    let w = f32((bingo << 16u) >> 28u);
    let c1 = vec4<f32>(x, y, z, w) / 7.0 * absmax;

    let a = f32((bingo << 12u) >> 28u);
    let b = f32((bingo << 8u) >> 28u);
    let c = f32((bingo << 4u) >> 28u);
    let d = f32((bingo >> 28u));
    let c2 = vec4<f32>(a, b, c, d) / 7.0 * absmax;
    return array<vec4<f32>, 2>(c1, c2);
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;
    let ND4 = N / 4u;
    let ND8 = N / 8u;
    let KD4 = K / 4u;
    let cRow = global_id.x;
    let cCol = global_id.y; 
    if (cRow < M && cCol < ND4) {
        var tmp0 = vec4<f32>(0.0);
        var tmp1 = vec4<f32>(0.0);
        for (var k = 0u; k < KD4; k++) {
          let a = A[cRow * KD4 + k];
          
          let bidx = (k * ND8 * 4u) + cCol;
          let b0 = unpack8x4snorm(B[bidx], {{ scale }}f);
          let b1 = unpack8x4snorm(B[bidx + (1u * ND8)], {{ scale }}f);
          let b2 = unpack8x4snorm(B[bidx + (2u * ND8)], {{ scale }}f);
          let b3 = unpack8x4snorm(B[bidx + (3u * ND8)], {{ scale }}f);

          tmp0 += vec4<f32>(a.x) * b0[0];
          tmp0 += vec4<f32>(a.y) * b1[0];
          tmp0 += vec4<f32>(a.z) * b2[0];
          tmp0 += vec4<f32>(a.w) * b3[0];

          tmp1 += vec4<f32>(a.x) * b0[1];
          tmp1 += vec4<f32>(a.y) * b1[1];
          tmp1 += vec4<f32>(a.z) * b2[1];
          tmp1 += vec4<f32>(a.w) * b3[1];
        }
        
        let cIdx = cRow * ND4 + cCol * 2u;
        C[cIdx] = tmp0; 
        C[cIdx + 1u] = tmp1; 
    }
}
