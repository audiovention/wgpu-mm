//Using SMEM
@group(0) @binding(0)
var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> C: array<vec4<f32>>;

var<workgroup> A_SHARED: array<vec4<f32>, {{ K / 4 }}>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{workgroup_size_z }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let M = {{ M }}u;
    let N = {{ N }}u;
    let K = {{ K }}u;

    //Each thread loads l elements into A_SHARED
    //Where l = ceil((vector_length / 4) / total_threads)
    for(var i = 0u; i < {{ loadPerThread }}u; i++) {
        let index = local_index + (i * {{ workgroup_size_x }}u);
        if (index < K / 4u){
            A_SHARED[index] =  A[index]; 
        }
    }
    workgroupBarrier();

    let ND4 = N / 4u;
    let cCol = global_id.x;

    if (cCol < ND4) {
        var tmp = vec4<f32>();
        for (var k = 0u; k < K / 4u; k++) {
          let a = A_SHARED[k];
          let bidx = k * N + cCol;
          tmp += vec4<f32>(a.x) * B[bidx]; 
          tmp += vec4<f32>(a.y) * B[bidx + (1u * ND4)]; 
          tmp += vec4<f32>(a.z) * B[bidx + (2u * ND4)];
          tmp += vec4<f32>(a.w) * B[bidx + (3u * ND4)];

        }
        C[cCol] = tmp; 
    }
}
