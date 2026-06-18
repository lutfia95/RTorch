use rtorch::{bf16, f16, statistics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let float16 = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
    let bfloat16 = [
        bf16::from_f32(2.0),
        bf16::from_f32(4.0),
        bf16::from_f32(6.0),
    ];

    println!("f16: {:?}", statistics(&float16)?);
    println!("bf16: {:?}", statistics(&bfloat16)?);
    println!("f32: {:?}", statistics(&[1.0_f32, 2.0, 3.0])?);
    println!("f64: {:?}", statistics(&[1.0_f64, 2.0, 3.0])?);
    println!("i64: {:?}", statistics(&[-10_i64, 0, 10])?);
    println!("u16: {:?}", statistics(&[10_u16, 20, 30])?);
    println!("bool: {:?}", statistics(&[true, false, true])?);

    Ok(())
}
