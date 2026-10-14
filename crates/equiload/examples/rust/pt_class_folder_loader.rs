use equiload::{DataLoader, DataLoaderConfig, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;

    println!("{:?}", dataset.class_names()?);
    println!("{}", dataset.len());

    let sample = dataset.get(0)?;
    println!("{:?} {}", sample.shape(), sample.label());

    let mut loader = DataLoader::with_config(
        &dataset,
        None,
        DataLoaderConfig {
            batch_size: 8,
            shuffle: true,
            drop_last: true,
        },
    )?;

    for batch in &mut loader {
        let batch = batch?;
        println!("{:?} {:?}", batch.shape(), batch.labels());
    }

    Ok(())
}
