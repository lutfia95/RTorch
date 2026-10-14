use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

use equiload::{
    bf16, f16, statistics, DataLoader, DataLoaderConfig, ScalarType, Tensor, TensorDataset,
    TransformConfig, TransformPipeline,
};

const VALUE_PRINT_LIMIT: usize = 32;

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempDataset {
    root: PathBuf,
}

impl TempDataset {
    fn from_single_pt(path: Option<PathBuf>) -> Result<Self, Box<dyn std::error::Error>> {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let root =
            env::temp_dir().join(format!("equiload-complete-example-{}-{id}", process::id()));
        let class_dir = root.join("example_class");
        fs::create_dir_all(&class_dir)?;

        let sample_path = class_dir.join("sample.pt");
        match path {
            Some(path) => {
                // EquiLoad loads .pt files through class-folder datasets. This copy turns one
                // user-provided tensor file into a one-class dataset without changing the input.
                fs::copy(path, &sample_path)?;
            }
            None => {
                // The default fixture is a Python torch.save tensor shaped [2, 3].
                fs::write(
                    &sample_path,
                    decode_base64(include_str!("../../../test_data/python_tensor.pt.base64"))?,
                )?;
            }
        }

        Ok(Self { root })
    }
}

impl Drop for TempDataset {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = env::args_os().nth(1).map(PathBuf::from);
    let temp_dataset = TempDataset::from_single_pt(input)?;

    println!("dataset root: {}", temp_dataset.root.display());

    demonstrate_pt_dataset(&temp_dataset.root)?;
    demonstrate_tensor_api()?;
    demonstrate_statistics_types()?;

    Ok(())
}

fn demonstrate_pt_dataset(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n== .pt file as a TensorDataset ==");

    let dataset = TensorDataset::from_class_folders(root)?;
    println!("dataset.len(): {}", dataset.len());
    println!("dataset.is_empty(): {}", dataset.is_empty());
    println!("dataset.class_names(): {:?}", dataset.class_names()?);

    let sample = dataset.get(0)?;
    println!("sample.shape(): {:?}", sample.shape());
    println!("sample.label(): {}", sample.label());
    print_data_view("sample.data() as f32", sample.shape(), || sample.data())?;
    println!("sample.statistics(): {:?}", sample.statistics()?);

    let tensor_from_sample = sample.tensor()?;
    print_tensor("sample.tensor()", &tensor_from_sample)?;

    // TransformConfig fields:
    // - to_float converts any tensor dtype to float32.
    // - resize, center_crop, normalize_mean, and normalize_std are for image-like
    //   tensors shaped C x H x W. They are left as None here so this example can
    //   run on any single tensor .pt file.
    let transform = TransformPipeline::new(&TransformConfig {
        to_float: true,
        resize: None,
        center_crop: None,
        normalize_mean: None,
        normalize_std: None,
    })?;

    let transformed_sample = dataset.get_with_transform(0, &transform)?;
    println!(
        "get_with_transform(...).shape(): {:?}",
        transformed_sample.shape()
    );
    print_data_view(
        "get_with_transform(...).data() as f32",
        transformed_sample.shape(),
        || transformed_sample.data(),
    )?;

    println!("\n== DataLoader ==");
    let mut loader = DataLoader::with_config(
        &dataset,
        Some(&transform),
        DataLoaderConfig {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
        },
    )?;
    println!("loader.len(): {}", loader.len());
    println!("loader.is_empty(): {}", loader.is_empty());

    if let Some(batch) = loader.next_batch()? {
        println!("next_batch().shape(): {:?}", batch.shape());
        println!("next_batch().labels(): {:?}", batch.labels());
        print_data_view("next_batch().data() as f32", batch.shape(), || batch.data())?;
        println!("next_batch().statistics(): {:?}", batch.statistics()?);
        print_tensor("next_batch().tensor()", &batch.tensor()?)?;
    }

    loader.reset()?;
    for batch in loader {
        let batch = batch?;
        println!("Iterator batch shape: {:?}", batch.shape());
    }

    // DataLoader::new(dataset, transforms, batch_size, shuffle) is the compact
    // constructor. DataLoader::with_config also exposes drop_last.
    let _compact_loader = DataLoader::new(&dataset, None, 1, false)?;

    Ok(())
}

fn demonstrate_tensor_api() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n== Tensor API ==");

    // Tensor::from_slice accepts Rust scalar slices implementing TensorElement:
    // bool, u8, i8, i16, i32, i64, usize/isize, f16, bf16, f32, and f64.
    // The unsigned u16/u32/u64 tensor dtypes require a LibTorch build with those
    // scalar types; statistics supports them portably by converting for reduction.
    let matrix = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2], false)?;
    print_tensor("Tensor::from_slice(f32 [2, 2])", &matrix)?;

    let zeros = Tensor::zeros(&[2, 2], ScalarType::F32, false)?;
    print_tensor("Tensor::zeros([2, 2], F32)", &zeros)?;

    let ones = Tensor::ones(&[2, 2], ScalarType::F32, false)?;
    print_tensor("Tensor::ones([2, 2], F32)", &ones)?;

    let cloned = matrix.try_clone()?;
    print_tensor("try_clone()", &cloned)?;

    let added = matrix.add(&ones)?;
    print_tensor("add()", &added)?;

    let multiplied = matrix.matmul(&ones)?;
    print_tensor("matmul()", &multiplied)?;

    let reshaped = matrix.reshape(&[4])?;
    print_tensor("reshape([4])", &reshaped)?;

    let summed = matrix.sum()?;
    print_tensor("sum()", &summed)?;

    let mut requires_grad = matrix.try_clone()?;
    requires_grad.set_requires_grad(true)?;
    println!("set_requires_grad(true): {}", requires_grad.requires_grad());

    let autograd_input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], true)?;
    let doubled = autograd_input.add(&autograd_input)?;
    let mut total = doubled.sum()?;
    total.backward()?;
    print_tensor(
        "grad() after sum(x + x).backward()",
        &autograd_input.grad()?,
    )?;

    println!("\nScalarType values exposed by EquiLoad:");
    for scalar_type in [
        ScalarType::Bool,
        ScalarType::U8,
        ScalarType::I8,
        ScalarType::I16,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::U16,
        ScalarType::U32,
        ScalarType::U64,
        ScalarType::F16,
        ScalarType::BF16,
        ScalarType::F32,
        ScalarType::F64,
    ] {
        match Tensor::zeros(&[1], scalar_type, false) {
            Ok(tensor) => {
                println!("{scalar_type:?}: {:?}", tensor.to_vec_f32()?);
            }
            Err(err) => {
                println!("{scalar_type:?}: not available in this LibTorch build ({err})");
            }
        }
    }

    Ok(())
}

fn demonstrate_statistics_types() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n== statistics(&[T]) supported Rust types ==");

    println!("bool: {:?}", statistics(&[true, false, true])?);
    println!("u8: {:?}", statistics(&[1_u8, 2, 3])?);
    println!("i8: {:?}", statistics(&[-1_i8, 0, 1])?);
    println!("i16: {:?}", statistics(&[-10_i16, 0, 10])?);
    println!("i32: {:?}", statistics(&[-10_i32, 0, 10])?);
    println!("i64: {:?}", statistics(&[-10_i64, 0, 10])?);
    println!("usize: {:?}", statistics(&[1_usize, 2, 3])?);
    println!("isize: {:?}", statistics(&[-1_isize, 0, 1])?);
    println!("u16: {:?}", statistics(&[10_u16, 20, 30])?);
    println!("u32: {:?}", statistics(&[10_u32, 20, 30])?);
    println!("u64: {:?}", statistics(&[10_u64, 20, 30])?);
    println!(
        "f16: {:?}",
        statistics(&[f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)])?
    );
    println!(
        "bf16: {:?}",
        statistics(&[
            bf16::from_f32(2.0),
            bf16::from_f32(4.0),
            bf16::from_f32(6.0)
        ])?
    );
    println!("f32: {:?}", statistics(&[1.0_f32, 2.0, 3.0])?);
    println!("f64: {:?}", statistics(&[1.0_f64, 2.0, 3.0])?);

    Ok(())
}

fn print_tensor(name: &str, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let shape = tensor.shape()?;
    let scalar_type = tensor.scalar_type()?;
    let numel = tensor.numel();
    let values = if numel <= VALUE_PRINT_LIMIT {
        format!("{:?}", tensor.to_vec_f32()?)
    } else {
        format!("<omitted {numel} values; VALUE_PRINT_LIMIT={VALUE_PRINT_LIMIT}>")
    };

    println!(
        "{name}: shape={:?}, scalar_type={:?}, numel={}, requires_grad={}, values_as_f32={}",
        shape,
        scalar_type,
        numel,
        tensor.requires_grad(),
        values
    );
    Ok(())
}

fn print_data_view<'a>(
    name: &str,
    shape: &[usize],
    data: impl FnOnce() -> Result<&'a [f32], equiload::Error>,
) -> Result<(), Box<dyn std::error::Error>> {
    match shape_numel(shape) {
        Some(numel) if numel <= VALUE_PRINT_LIMIT => {
            println!("{name}: {:?}", data()?);
        }
        Some(numel) => {
            println!("{name}: <omitted {numel} values; VALUE_PRINT_LIMIT={VALUE_PRINT_LIMIT}>");
        }
        None => {
            println!("{name}: <omitted; shape element count overflowed usize>");
        }
    }
    Ok(())
}

fn shape_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
}

fn decode_base64(input: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut output = Vec::new();
    let mut buffer = 0_u32;
    let mut bits = 0_u8;

    for byte in input.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
        if byte == b'=' {
            break;
        }

        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return Err(format!("invalid base64 byte: {byte}").into()),
        };

        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= (1_u32 << bits) - 1;
        }
    }

    Ok(output)
}
