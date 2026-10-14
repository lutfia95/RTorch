use equiload::{DataLoader, DataLoaderConfig, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_hdf5_file("test_data/train.h5", "positions")?;

    let sample = dataset.get(0)?;
    println!("{:?} {}", sample.shape(), sample.label());

    let mut loader = DataLoader::with_config(
        &dataset,
        None,
        DataLoaderConfig {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
        },
    )?;

    for batch in &mut loader {
        let batch = batch?;
        println!("{:?} {:?}", batch.shape(), batch.labels());
    }

    Ok(())
}
