use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    (M, N, K)
}

pub fn sgemm_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_1.wgsl",
        include_str!("../shaders/sgemm/sgemm_1.wgsl"),
    )
    .unwrap();
    let workgroup_size_x = 16;
    let workgroup_size_y = 16;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_1.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn sgemm_1v(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_1v.wgsl",
        include_str!("../shaders/sgemm/sgemm_1v.wgsl"),
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
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_1v.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn sgemm_2(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_2.wgsl",
        include_str!("../shaders/sgemm/sgemm_2.wgsl"),
    )
    .unwrap();
    let workgroup_size_x = 256;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(M, 16) as _, Workload::ceil(N, 16) as _, 1),
        WorkgroupSize(workgroup_size_x, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_2.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn sgemm_3(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_3.wgsl",
        include_str!("../shaders/sgemm/sgemm_3.wgsl"),
    )
    .unwrap();
    let BLOCKSIZE = 16;
    context.insert("BLOCKSIZE", &BLOCKSIZE);
    let workgroup_size_x = (BLOCKSIZE * BLOCKSIZE) / 4;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, BLOCKSIZE) as _,
            Workload::ceil(N, BLOCKSIZE) as _,
            1,
        ),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_3.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn sgemm_4(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_4.wgsl",
        include_str!("../shaders/sgemm/sgemm_4.wgsl"),
    )
    .unwrap();
    let BM = 16;
    let BN = 16;
    let BK = 8;
    let TM = 2;

    context.insert("BM", &BM);
    context.insert("BN", &BN);
    context.insert("BK", &BK);
    context.insert("TM", &TM);

    let workgroup_size_x = (BM * BN) / TM;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, BN) as _, Workload::ceil(M, BM) as _, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_4.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn sgemm_5(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "sgemm_5.wgsl",
        include_str!("../shaders/sgemm/sgemm_5.wgsl"),
    )
    .unwrap();
    let BM = 32;
    let BN = 32;
    let BK = 16;
    let TM = 4;
    let TN = 4;

    context.insert("BM", &BM);
    context.insert("BN", &BN);
    context.insert("BK", &BK);
    context.insert("TM", &TM);
    context.insert("TN", &TN);

    let workgroup_size_x = (BM * BN) / (TM * TN);
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, BN) as _, Workload::ceil(M, BM) as _, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    println!("workload: {:?}", workload);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("sgemm_5.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

pub fn sgemm_tf(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("tfjs.wgsl", include_str!("../shaders/sgemm/tfjs.wgsl"))
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

    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);

    let shader = tera.render("tfjs.wgsl", &context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::test_harness;

    use super::*;

    macro_rules! sgemm_test {
        ($test_name:ident, $sgemm_function:ident) => {
            #[tokio::test]
            pub async fn $test_name() {
                let _ = env_logger::builder().is_test(true).try_init();
                let mut tera = tera::Tera::default();
                let mut context = tera::Context::new();
                let dims = insert_matrix_dims(&mut context);
                let (workload, shader) = $sgemm_function(&mut tera, &mut context);
                test_harness(workload, shader, dims, false).await;
            }
        };
    }

    sgemm_test!(test_sgemm_1, sgemm_1);
    sgemm_test!(test_sgemm_1v, sgemm_1v);
    sgemm_test!(test_sgemm_2, sgemm_2);
    sgemm_test!(test_sgemm_3, sgemm_3);
    sgemm_test!(test_sgemm_4, sgemm_4);
    sgemm_test!(test_sgemm_5, sgemm_5);
    sgemm_test!(test_sgemm_tf, sgemm_tf);
}
