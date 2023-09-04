use tera::{Context, Tera};

use crate::{WorkgroupCount, WorkgroupSize, Workload};

const M: usize = 1;
const N: usize = 51868;
const K: usize = 384;
pub const ABSMAX: f32 = 0.2; //Data ranges from -10 to 10, divide by 50

pub fn insert_matrix_dims(context: &mut Context) -> (usize, usize, usize) {
    context.insert("M", &M);
    context.insert("N", &N);
    context.insert("K", &K);
    context.insert("MD4", &(M / 4));
    context.insert("ND4", &(N / 4));
    context.insert("KD4", &(K / 4));
    (M, N, K)
}

pub fn hgemv_1(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "hgemv_1.wgsl",
        include_str!("../shaders/hgemv/hgemv_1.wgsl"),
    )
    .unwrap();
    let workgroup_size_x = 16;
    let workgroup_size_y = 1;
    let workgroup_size_z = 1;
    let colPerThread = 2;

    let wgs = WorkgroupSize(workgroup_size_x as _, workgroup_size_y, workgroup_size_z);

    let requiredGroups = Workload::ceil(N / (colPerThread * 4), wgs.total() as usize);

    let workload = Workload::new(WorkgroupCount(requiredGroups as _, 1, 1), wgs);
    context.insert("colPerThread", &colPerThread);
    context.insert("workgroup_size_x", &workload.size().0);
    context.insert("workgroup_size_y", &workload.size().1);
    context.insert("workgroup_size_z", &workload.size().2);
    let shader = tera.render("hgemv_1.wgsl", context).unwrap();
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::quant::Quantization;
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
                test_harness(workload, shader, dims, Quantization::Float16).await;
            }
        };
    }

    gemv_test!(test_hgemv_1, hgemv_1);
}
