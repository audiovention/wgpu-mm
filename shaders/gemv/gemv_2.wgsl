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
    let ND4 = {{ ND4 }}u;
    let KD4 = {{ KD4 }}u;

    let cCol = global_id.x * {{ colPerThread }}u;  
    if (cCol < ND4) {
        var tmp = mat{{ colPerThread }}x4<f32>();
        for (var k = 0u; k < KD4; k++) {
          let a = A[k];
          let bidx = k * N + cCol;
            
          {%- for i in range(end=colPerThread) %}
              tmp[{{ i }}] += vec4<f32>(a.x) * B[bidx + {{ i }}u]; 
              tmp[{{ i }}] += vec4<f32>(a.y) * B[bidx + ND4 + {{ i }}u]; 
              tmp[{{ i }}] += vec4<f32>(a.z) * B[bidx + (2u * ND4) + {{ i }}u];
              tmp[{{ i }}] += vec4<f32>(a.w) * B[bidx + (3u * ND4) + {{ i }}u];
          {% endfor -%}
        }

        {% for i in range(end=colPerThread) %}
            C[cCol + {{ i }}u] = tmp[{{ i }}];
        {%- endfor %}
    }
}
