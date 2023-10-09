use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    context.insert("MD4", &(M / 4));
    context.insert("ND4", &(N / 4));
    context.insert("KD4", &(K / 4));
    (M, N, K)
}
const M: usize = 1;
const N: usize = 51868;
const K: usize = 384;
pub const ABSMAX: f32 = 0.2; //Data ranges from -10 to 10, divide by 50

pub fn sint8_gemv_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("qgemv_1.wgsl", include_str!("../shaders/gemv/qgemv_1.wgsl"))
        .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let colPerThread = 1;

    let wgs = WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z);

    let requiredGroups = Workload::ceil(N / (colPerThread * 4), wgs.total() as usize);

    let workload = Workload::new(WorkgroupCount(requiredGroups as _, 1, 1), wgs);
    println!("Workload: {:?}", workload);
    context.insert("colPerThread", &colPerThread);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    context.insert("scale", &ABSMAX);
    let shader = tera.render("qgemv_1.wgsl", context).unwrap();
    println!("Shader: {}", shader);
    (workload, shader)
}

pub fn sint4_gemv_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("qgemv_1.wgsl", include_str!("../shaders/gemv/qgemv_1.wgsl"))
        .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let colPerThread = 1;

    let wgs = WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z);

    let requiredGroups = Workload::ceil(N / (colPerThread * 4), wgs.total() as usize);

    let workload = Workload::new(WorkgroupCount(requiredGroups as _, 1, 1), wgs);
    println!("Workload: {:?}", workload);
    context.insert("colPerThread", &colPerThread);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    context.insert("scale", &ABSMAX);
    let shader = tera.render("qgemv_1.wgsl", context).unwrap();
    println!("Shader: {}", shader);
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qgemv::insert_matrix_dims;
    use crate::quant::Quantization;
    use crate::test_harness;

    #[tokio::test]
    pub async fn test_qgemv_1() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        let dims = insert_matrix_dims(&mut context);
        let (workload, shader) = sint8_gemv_1(&mut tera, &mut context);
        test_harness(workload, shader, dims, Quantization::SInt8).await;
    }
}
