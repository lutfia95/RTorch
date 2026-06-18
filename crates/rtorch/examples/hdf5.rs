use std::env;

use rtorch::{DataLoader, DataLoaderConfig, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "test_data/train.h5".to_string());
    let dataset_name = env::args()
        .nth(2)
        .unwrap_or_else(|| "positions".to_string());

    let dataset = TensorDataset::from_hdf5_file(&path, &dataset_name)?;
    println!("dataset: {:?}", dataset.get(4));
    println!("samples: {}", dataset.len());
    println!("classes: {:?}", dataset.class_names()?);

    let sample = dataset.get(0)?;
    println!("first sample shape: {:?}", sample.shape());
    println!("first sample label: {:?}", sample.label());

    

    let mut loader = DataLoader::with_config(
        &dataset,
        None,
        DataLoaderConfig {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
        },
    )?;
    println!("batches: {}", loader.len());

    if let Some(batch) = loader.next().transpose()? {
        println!("first batch shape: {:?}", batch.shape());
        println!("first batch labels: {:?}", batch.labels());
        println!("first batch statistics: {:?}", batch.statistics()?);
    }

    Ok(())
}
