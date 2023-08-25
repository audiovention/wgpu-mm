use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

const M: usize = 1;
const N: usize = 1024;
const K: usize = 1024;
pub const ABSMAX: f32 = 0.2; //Data ranges from -10 to 10, divide by 50

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    (M, N, K)
}

pub fn gemv_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("gemv_1.wgsl", include_str!("../shaders/gemv/gemv_1.wgsl"))
        .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let wgs = WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z);
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, workgroup_size_x) as _,
            Workload::ceil(N, (workgroup_size_y * 4) as _) as _,
            1,
        ),
        wgs,
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("gemv_1.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn gemv_tfjs(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("tfjs.wgsl", include_str!("../shaders/gemv/tfjs.wgsl"))
        .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 8;
    let workgroup_size_z = 1;
    let wgs = WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z);
    let workload = Workload::new(
        WorkgroupCount(
            Workload::ceil(M, workgroup_size_x) as _,
            Workload::ceil(N, (workgroup_size_y * 4) as _) as _,
            1,
        ),
        wgs,
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("tfjs.wgsl", &context).unwrap();
    (workload, shader)
}

pub fn qgemv_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template("qgemv_1.wgsl", include_str!("../shaders/gemv/qgemv_1.wgsl"))
        .unwrap();
    let workgroup_size_x = 8;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let workload = Workload::new(
        WorkgroupCount(Workload::ceil(N, workgroup_size_x * 4) as _, 1, 1),
        WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z),
    );
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    context.insert("absmax", &ABSMAX);
    let shader = tera.render("qgemv_1.wgsl", &context).unwrap();
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::test_harness;

    use super::*;

    macro_rules! gemv_test {
        ($test_name:ident, $gemv_function:ident) => {
            #[tokio::test]
            pub async fn $test_name() {
                let _ = env_logger::builder().is_test(true).try_init();
                let mut tera = tera::Tera::default();
                let mut context = tera::Context::new();
                let dims = insert_matrix_dims(&mut context);
                let (workload, shader) = $gemv_function(&mut tera, &mut context);
                test_harness(workload, shader, dims, false).await;
            }
        };
    }

    #[tokio::test]
    pub async fn test_qgemv_1() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut tera = tera::Tera::default();
        let mut context = tera::Context::new();
        let dims = insert_matrix_dims(&mut context);
        let (workload, shader) = qgemv_1(&mut tera, &mut context);
        test_harness(workload, shader, dims, true).await;
    }

    gemv_test!(test_gemv_1, gemv_1);
    gemv_test!(test_gemv_tfjs, gemv_tfjs);
}
