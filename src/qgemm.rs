use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const ABSMAX: f32 = 0.2;

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    (M, N, K)
}

pub fn qgemm_int8_naive(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "qgemm_int8.wgsl",
        include_str!("../shaders/qgemm/qgemm_int8.wgsl"),
    )
    .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, workgroup_size_x as _) as _,
            Workload::ceil(N, (workgroup_size_y * 4) as _) as _,
            1,
        ),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    context.insert("scale", &ABSMAX);
    let shader = tera.render("qgemm_int8.wgsl", context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn qgemm_int8_tiled(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "qgemm_tiled.wgsl",
        include_str!("../shaders/qgemm/qgemm_int8_tiled.wgsl"),
    )
    .unwrap();

    let TILE_DIM = 32;
    let ROW_PER_THREAD = 4;
    let workgroup_size = WorkgroupSize((TILE_DIM / 4) as _, (TILE_DIM / ROW_PER_THREAD) as _, 1);
    let group_x = Workload::ceil(N, TILE_DIM);
    let group_y = Workload::ceil(M, TILE_DIM);

    let workgroup_count = WorkgroupCount(group_x as _, group_y as _, 1);
    let workload = Workload::new(workgroup_count, workgroup_size);

    let aShape = vec![1, M, K];
    let aShapeStrides = vec![M * K, M];
    let bShape = vec![1, K, N];
    let bShapeStrides = vec![K * N, N];
    let outShape = vec![1, M, N];
    let outShapeStrides = vec![M * N, M];
    let dimAOuter = M;
    let dimBOuter = N;
    let dimInner = K;

    context.insert("TILE_DIM", &TILE_DIM);
    context.insert("ROW_PER_THREAD", &ROW_PER_THREAD);
    context.insert("aShape", &aShape);
    context.insert("aShapeStrides", &aShapeStrides);
    context.insert("bShape", &bShape);
    context.insert("bShapeStrides", &bShapeStrides);
    context.insert("outShape", &outShape);
    context.insert("outShapeStrides", &outShapeStrides);
    context.insert("dimAOuter", &dimAOuter);
    context.insert("dimBOuter", &dimBOuter);
    context.insert("dimInner", &dimInner);
    context.insert("scale", &ABSMAX);

    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);

    let shader = tera.render("qgemm_tiled.wgsl", context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn qgemm_int4(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "qgemm_int4.wgsl",
        include_str!("../shaders/qgemm/qgemm_int4.wgsl"),
    )
    .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, workgroup_size_x as _) as _,
            Workload::ceil(N, (workgroup_size_y * 8) as _) as _,
            1,
        ),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    context.insert("scale", &ABSMAX);
    let shader = tera.render("qgemm_int4.wgsl", context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::quant::Quantization;
    use crate::test_harness;

    use super::*;

    macro_rules! qgemm_test {
        ($test_name:ident, $qgemm_function:ident, $quantization:expr) => {
            #[tokio::test]
            pub async fn $test_name() {
                let _ = env_logger::builder().is_test(true).try_init();
                let mut tera = tera::Tera::default();
                let mut context = tera::Context::new();
                let dims = insert_matrix_dims(&mut context);
                let (workload, shader) = $qgemm_function(&mut tera, &mut context);
                test_harness(workload, shader, dims, $quantization).await;
            }
        };
    }

    qgemm_test!(test_qgemm_int8_naive, qgemm_int8_naive, Quantization::SInt8);
    qgemm_test!(test_qgemm_int8_tiled, qgemm_int8_tiled, Quantization::SInt8);
    qgemm_test!(test_qgemm_int4, qgemm_int4, Quantization::SInt4);
}
