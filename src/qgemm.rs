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

pub fn qgemm_1v(tera: &mut Tera, context: &mut Context) -> (Workload, String) {
    tera.add_raw_template(
        "qgemm_1v.wgsl",
        include_str!("../shaders/qgemm/qgemm_1v.wgsl"),
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
    let shader = tera.render("qgemm_1v.wgsl", context).unwrap();
    println!("shader: {}", shader);
    (workload, shader)
}

#[cfg(test)]
mod tests {
    use crate::quant::Quantization;
    use crate::test_harness;

    use super::*;

    macro_rules! qgemm_test {
        ($test_name:ident, $qgemm_function:ident) => {
            #[tokio::test]
            pub async fn $test_name() {
                let _ = env_logger::builder().is_test(true).try_init();
                let mut tera = tera::Tera::default();
                let mut context = tera::Context::new();
                let dims = insert_matrix_dims(&mut context);
                let (workload, shader) = $qgemm_function(&mut tera, &mut context);
                test_harness(workload, shader, dims, Quantization::SInt8).await;
            }
        };
    }

    //qgemm_test!(test_qgemm_1, qgemm_1);
    qgemm_test!(test_qgemm_1v, qgemm_1v);
}
