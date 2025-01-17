// Dumped WGSL:

  struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
  struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

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
  fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
    let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
  }
  fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
    let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
  }

  // NaN defination in IEEE 754-1985 is :
  //   - sign = either 0 or 1.
  //   - biased exponent = all 1 bits.
  //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
  // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
  fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
  }
  fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
    let floatToUint: vec4<u32> = bitcast<vec4<u32>>(val);
    return (floatToUint & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
  }



      var<private> localId: vec3<u32>;
      var<private> localIndex: u32;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;
      var<private> workgroupId: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
          return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
                workgroupId.y * numWorkgroups.x + workgroupId.x) * 64u +
                localIndex);
        
      }
    
struct Uniforms { NAN : f32, INFINITY : f32, aShape : vec3<i32>, aShapeStrides: vec2<i32>, bShape : vec3<i32>, bShapeStrides: vec2<i32>, outShape : vec3<i32>, 
         outShapeStrides: vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,};

      @group(0) @binding(0) var<storage, read_write> result: array<vec4<f32>>;
    

      @group(0) @binding(1) var<storage, read> A: array<vec4<f32>>;
        

      @group(0) @binding(2) var<storage, read> B: array<vec4<f32>>;
        

      @group(0) @binding(3) var<uniform> uniforms: Uniforms;
      
  fn isinf(val: f32) -> bool {
    return abs(val) == uniforms.INFINITY;
  }


    fn getCoordsFromIndex(index : i32) -> vec3<i32> {
      var index2 = index;let d0 = index2 / uniforms.outShapeStrides.x; index2 = index2 - d0 * uniforms.outShapeStrides.x;let d1 = index2 / uniforms.outShapeStrides.y; let d2 = index2 - d1 * uniforms.outShapeStrides.y;
      return vec3<i32>(d0,d1,d2);
    }
  
fn getOutputCoords() -> vec3<i32> {
  let d2 = i32(globalId[0]);let d1 = i32(globalId[1]);let d0 = i32(globalId[2]);
return vec3<i32>(d0,d1,d2); }

        fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
          return dot(coords, vec3<i32>(uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, 1));
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
      var index2 = index;let d0 = index2 / uniforms.aShapeStrides.x; index2 = index2 - d0 * uniforms.aShapeStrides.x;let d1 = index2 / uniforms.aShapeStrides.y; let d2 = index2 - d1 * uniforms.aShapeStrides.y;
      return vec3<i32>(d0,d1,d2);
    }
  

    fn getBCoordsFromIndex(index : i32) -> vec3<i32> {
      var index2 = index;let d0 = index2 / uniforms.bShapeStrides.x; index2 = index2 - d0 * uniforms.bShapeStrides.x;let d1 = index2 / uniforms.bShapeStrides.y; let d2 = index2 - d1 * uniforms.bShapeStrides.y;
      return vec3<i32>(d0,d1,d2);
    }
  

    fn getA(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
      return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(d0,d1,d2),
        uniforms.aShape) / 4]);
    }
   
  fn getAByOutputIndex(globalIndex : i32) -> vec4<f32> {
    var coords = getCoordsFromIndex(globalIndex);
    
    return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), uniforms.aShape) / 4]);
  }

  fn getAByOutputCoords(coordsIn : vec3<i32>) -> vec4<f32> {
    var coords = coordsIn;
    
    return vec4<f32>(A[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), uniforms.aShape) / 4]);
  }


    fn getB(d0 : i32, d1 : i32, d2 : i32) -> vec4<f32> {
      return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(d0,d1,d2),
        uniforms.bShape) / 4]);
    }
   
  fn getBByOutputIndex(globalIndex : i32) -> vec4<f32> {
    var coords = getCoordsFromIndex(globalIndex);
    
    return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), uniforms.bShape) / 4]);
  }

  fn getBByOutputCoords(coordsIn : vec3<i32>) -> vec4<f32> {
    var coords = coordsIn;
    
    return vec4<f32>(B[getIndexFromCoords3D(vec3<i32>(coords.x, coords.y, coords.z), uniforms.bShape) / 4]);
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
    
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      
      
      
      
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  
      
  var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 32>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;

  
        fn main()
       {
    let localRow = i32(localId.y);
    let tileRow = localRow * 4;
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * 4;
    let globalCol = i32(globalId.x) * 4;
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];
    let globalRowStart = i32(workgroupId.y) * 32;

    let numTiles = (uniforms.dimInner - 1) / 32 + 1;
    var kStart = 0;

    var acc: array<vec4<f32>, 4>;

    // Loop over shared dimension.
    let tileRowB = localRow * 4;
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * 4);
        
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < 4; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + 32;
        workgroupBarrier();

        // Compute acc values for a single thread.
        
      for (var k = 0; k < 8; k++) {
        let BCached0 = mm_Bsub[k * 4 + 0][tileCol];let BCached1 = mm_Bsub[k * 4 + 1][tileCol];let BCached2 = mm_Bsub[k * 4 + 2][tileCol];let BCached3 = mm_Bsub[k * 4 + 3][tileCol];
        for (var i = 0; i < 4; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < 4; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }
    

     
  @compute @workgroup_size(8, 8, 1)

      fn _start(@builtin(local_invocation_id) LocalId : vec3<u32>,
                @builtin(global_invocation_id) GlobalId : vec3<u32>,
                @builtin(local_invocation_index) LocalIndex: u32,
                @builtin(workgroup_id) WorkgroupId : vec3<u32>,
                @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
        localId = LocalId;
        localIndex = LocalIndex;
        globalId = GlobalId;
        numWorkgroups = NumWorkgroups;
        workgroupId = WorkgroupId;
        main();;
      }
    
