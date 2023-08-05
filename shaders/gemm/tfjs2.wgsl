  // Checks whether coordinates lie within the bounds of the shape.
fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) && all(coord < shape);
}
fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) && all(coord < shape);
}
fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) && all(coord < shape);
}

fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
    return coord;
}

fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return dot(coords, vec2<i32>(shape.y, 1));
}

fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
}

fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return dot(coords, vec4<i32>(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
}

fn getCoordsFromIndex(index : i32) -> vec3<i32> {
    var index2 = index;
    let d0 = index2 / {{ outShapeStrides[0] }};
    index2 = index2 - d0 * {{ outShapeStrides[0] }}; 
    let d1 = index2 / {{ outShapeStrides[1] }};
    let d2 = index2 - d1 * {{ outShapeStrides[1] }}; 
    return vec3<i32>(d0,d1,d2);
}
  
fn getOutputCoords() -> vec3<i32> {
  let d2 = i32(globalId[0]);let d1 = i32(globalId[1]);let d0 = i32(globalId[2]);
    return vec3<i32>(d0,d1,d2); 
}

fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
  return dot(coords, vec3<i32>({{ outShapeStrides[0] }}, {{ outShapeStrides[1] }}, 1));
}
        
fn setOutputAtIndex(flatIndex : i32, value : vec4<f32>) {
      result[flatIndex] = vec4<f32>(value);
    }

fn setOutputAtIndexI32(flatIndex : i32, value : vec4<i32>) {
    result[flatIndex] = vec4<f32>(value);
}

fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, value : vec4<f32>) {
    let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
    setOutputAtIndex(flatIndex / 4, value);
}
fn setOutputAtCoordsI32(d0 : i32, d1 : i32, d2 : i32, value : vec4<i32>) {
    let flatIndex = getOutputIndexFromCoords(vec3<i32>(d0, d1, d2));
    setOutputAtIndexI32(flatIndex / 4, value);
}
    

fn getACoordsFromIndex(index : i32) -> vec3<i32> {
    var index2 = index;
    let d0 = index2 / {{ aShapeStrides[0] }}; 
    index2 = index2 - d0 * {{ aShapeStrides[0] }}; 
    let d1 = index2 / {{ aShapeStrides[1] }};
    let d2 = index2 - d1 * {{ aShapeStrides[1] }}; 
    return vec3<i32>(d0,d1,d2);
}

fn getBCoordsFromIndex(index : i32) -> vec3<i32> {
    var index2 = index;
    let d0 = index2 / {{ bShapeStrides[0] }}; 
    index2 = index2 - d0 * {{ bShapeStrides[0] }}; 
    let d1 = index2 / {{ bShapeStrides[1] }}; 
    let d2 = index2 - d1 * {{ bShapeStrides[1] }}; 
    return vec3<i32>(d0,d1,d2);
}

fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
    return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(d0,d1,d2), vec3<i32>({{ aShape.0 }}, {{ aShape.1 }} , {{ aShape.2 }})) / 4]);
}
   
fn getAByOutputIndex(globalIndex : i32) -> vec4<f32> {
    var coords = getCoordsFromIndex(globalIndex);
    return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), vec3<i32>({{ aShape.0 }}, {{ aShape.1 }} , {{ aShape.2 }})) / 4]);
}

fn getAByOutputCoords(coordsIn : vec3<i32>) -> vec4<f32> {
    var coords = coordsIn;
    return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), vec3<i32>({{ aShape.0 }}, {{ aShape.1 }} , {{ aShape.2 }})) / 4]);
}

fn getB(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
    return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(d0,d1,d2), vec3<i32>({{ bShape.0 }}, {{ bShape.1 }} , {{ bShape.2 }})) / 4]);
}
   
fn getBByOutputIndex(globalIndex : i32) -> vec4<f32> {
    var coords = getCoordsFromIndex(globalIndex);

    return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), vec3<i32>({{ bShape.0 }}, {{ bShape.1 }} , {{ bShape.2 }})) / 4]);
}

fn getBByOutputCoords(coordsIn : vec3<i32>) -> vec4<f32> {
    var coords = coordsIn;

    return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), vec3<i32>({{ bShape.0 }}, {{ bShape.1 }} , {{ bShape.2 }})) / 4]);
}
  
fn mm_readA(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    value = getA(batch, row, col);
    return value;
}

fn mm_readB(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    value = getB(batch, row, col);
    return value;
}
  
fn mm_write(batch: i32, row: i32, col: i32, valueIn: vec4<f32>) {
    var value = valueIn;
    let coords = vec3<i32>(batch, row, col);
    setOutputAtCoords(coords[0], coords[1], coords[2], value);
}
      


var<private> localId: vec3<u32>;
var<private> localIndex: u32;
var<private> globalId: vec3<u32>;
var<private> numWorkgroups: vec3<u32>;
var<private> workgroupId: vec3<u32>;


@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;

@group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;

@group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>; //32x32 tile
var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>; //32x32 tile
  
@compute @workgroup_size(8, 8, 1) 
fn main(@builtin(local_invocation_id) localId : vec3<u32>,
        @builtin(global_invocation_id) globalId : vec3<u32>,
        @builtin(workgroup_id) workgroupId : vec3<u32>) {

    let localRow = i32(localId.y);
    let tileRow = localRow * 4; //rowPerThread
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * 4; //rowPerThread
    let globalCol = i32(globalId.x) * 4; //colPerThread

    let batch = i32(globalId.z);
    let batchA = batch % {{ aShape[0] }}; 
    let batchB = batch % {{ bShape[0] }};

    let numTiles = ({{ dimInner }} - 1) / 32 + 1;
    var kStart = 0;

    var acc: array<vec4<f32>, 4>;


    var ACached: vec4<f32>;
    var BCached0: vec4<f32>;
    var BCached1: vec4<f32>;
    var BCached2: vec4<f32>;
    var BCached3: vec4<f32>;
    // Loop over shared dimension.
    {% for tile in range(end=32) %}
        // Load one tile of A into local memory.
        {% for innerRow in range(end=4) %}
            mm_Asub[tileRow + {{ innerRow }}][tileCol] = mm_readA(batchA, globalRow + {{ innerRow }}, kStart + tileCol * 4);
        {% endfor %}

        // Load one tile of B into local memory.
        {% for innerRow in range(end=4) %}
            mm_Bsub[tileRow + {{ innerRow }}][tileCol] = mm_readB(batchB, kStart + tileRow + {{ innerRow }}, globalCol);
        {% endfor %}
        kStart = kStart + 32;
        workgroupBarrier();

        // Compute acc values for a single thread.
        {% for k in range(end=8) %}
            BCached0 = mm_Bsub[{{ k }} * 4 + 0][tileCol];
            BCached1 = mm_Bsub[{{ k }} * 4 + 1][tileCol];
            BCached2 = mm_Bsub[{{ k }} * 4 + 2][tileCol];
            BCached3 = mm_Bsub[{{ k }} * 4 + 3][tileCol];

            {% for i in range(end=4) %}
              ACached = mm_Asub[tileRow + {{i}}][{{ k }}];
              acc[{{i}}] = fma(BCached0, vec4<f32>(ACached[0]), acc[{{i}}]);
              acc[{{i}}] = fma(BCached1, vec4<f32>(ACached[1]), acc[{{i}}]);
              acc[{{i}}] = fma(BCached2, vec4<f32>(ACached[2]), acc[{{i}}]);
              acc[{{i}}] = fma(BCached3, vec4<f32>(ACached[3]), acc[{{i}}]);
            {% endfor %}
        {% endfor %}
        workgroupBarrier();
    {% endfor %}

    {% for innerRow in range(end=4) %}
        mm_write(batch, globalRow + {{ innerRow }}, globalCol, acc[{{ innerRow }}]);
    {% endfor %}
  }

