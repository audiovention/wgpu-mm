fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
}

struct Uniforms {  aShape : vec3<i32>, aShapeStrides: vec2<i32>, bShape : vec3<i32>, bShapeStrides: vec2<i32>, outShape : vec3<i32>, 
     outShapeStrides: vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,};


  @group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;

  @group(0) @binding(1) var<storage, read> B: array<vec4<f32>>;

  @group(0) @binding(2) var<storage, read_write> result: array<vec4<f32>>;

    fn getCoordsFromIndex(index : i32) -> vec3<i32> {
      var index2 = index;let d0 = index2 / uniforms.outShapeStrides.x; index2 = index2 - d0 * uniforms.outShapeStrides.x;let d1 = index2 / uniforms.outShapeStrides.y; let d2 = index2 - d1 * uniforms.outShapeStrides.y;
      return vec3<i32>(d0,d1,d2);
    }

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
    
    if(row < uniforms.aShape[1] && col < uniforms.aShape[2])
    {
      
      value = getA(batch, row, col);

    
    }
    
    return value;
  }

  fn mm_readB(batch: i32, row: i32, col: i32) -> vec4<f32> {
    var value = vec4<f32>(0.0);
    value = getB(batch, row, col);
    return value;
  }
  
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: vec4<f32>) {
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      
      
      
      
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  
      
  var<workgroup> mm_Asub : array<array<vec4<f32>, 8>, 8>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, 8>, 32>;

  
  @compute @workgroup_size(8, 8, 1)
      fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
                @builtin(global_invocation_id) GlobalId : vec3<u32>,
                @builtin(local_invocation_index) LocalIndex: u32,
                @builtin(workgroup_id) WorkgroupId : vec3<u32>,
                @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
    let localRow = i32(localId.y);
    let tileRow = localRow * 1;
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * 1;
    let globalCol = i32(globalId.x) * 4;
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];
    let globalRowStart = i32(workgroupId.y) * 8;

    let numTiles = (uniforms.dimInner - 1) / 32 + 1;
    var kStart = 0;

    var acc: array<vec4<f32>, 1>;

    // Loop over shared dimension.
    let tileRowB = localRow * 4;
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < 1; innerRow++) {
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
        for (var i = 0; i < 1; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = fma(BCached0, vec4<f32>(ACached[0]), acc[i]);acc[i] = fma(BCached1, vec4<f32>(ACached[1]), acc[i]);acc[i] = fma(BCached2, vec4<f32>(ACached[2]), acc[i]);acc[i] = fma(BCached3, vec4<f32>(ACached[3]), acc[i]);
        }
      }
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < 1; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
      }
    
