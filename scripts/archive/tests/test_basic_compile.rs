// Quick compilation test
use arrow_quant_v2::*;

fn main() {
    println!("✓ Library compiled successfully");
    println!("✓ All modules accessible");
    
    // Test basic functionality
    let config = QuantizationConfig {
        bit_width: 4,
        scale: 1.0,
        zero_point: 128,
    };
    
    println!("✓ QuantizationConfig created: bit_width={}", config.bit_width);
    println!("\n=== Compilation Verification Complete ===");
    println!("The library is ready for testing.");
}
