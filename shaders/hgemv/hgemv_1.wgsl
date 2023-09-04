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
    let ND4 = {{ ND4 }}u;
    let KD4 = {{ KD4 }}u;

    let cCol = global_id.x * {{ colPerThread }}u; 
    if (cCol < ND4) {
        var tmp = mat{{colPerThread}}x4<f32>();
        for (var k = 0u; k < KD4; k++) {
          let a = A[k];
          let bidx = (k * ND4 * 4u) + cCol;

          {% for i in range(end=colPerThread) %}
            tmp[{{ i }}] += vec4<f32>(a.x) * unpack4x16float(B[bidx + {{ i }}u]);
            tmp[{{ i }}] += vec4<f32>(a.y) * unpack4x16float(B[bidx + ({{ i }}u + (1u * ND4))]);
            tmp[{{ i }}] += vec4<f32>(a.z) * unpack4x16float(B[bidx + ({{ i }}u + (2u * ND4))]);
            tmp[{{ i }}] += vec4<f32>(a.w) * unpack4x16float(B[bidx + ({{ i }}u + (3u * ND4))]);
          {% endfor %}
        }
        {% for i in range(end=colPerThread) %}
          C[cCol + {{ i }}u] = tmp[{{ i }}];
        {% endfor %} 
    }
}
