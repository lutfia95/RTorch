use equiload::{DataLoader, TensorDataset, TransformConfig, TransformPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;
    let transforms = TransformPipeline::new(&TransformConfig {
        to_float: true,
        resize: Some((256, 256)),
        center_crop: Some((224, 224)),
        normalize_mean: Some(vec![0.485, 0.456, 0.406]),
        normalize_std: Some(vec![0.229, 0.224, 0.225]),
    })?;

    let mut loader = DataLoader::new(&dataset, Some(&transforms), 4, true)?;

    while let Some(batch) = loader.next_batch()? {
        println!(
            "shape={:?} labels={:?} statistics={:?}",
            batch.shape(),
            batch.labels(),
            batch.statistics()?
        );
    }

    Ok(())
}
