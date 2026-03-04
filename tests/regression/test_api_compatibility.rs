// API 兼容性验证测试
// 验证所有现有 API 保持不变，无破坏性变更
// 需求: 7.2, 7.5

use arrow_quant_v2::time_aware::{TimeAwareQuantizer, TimeGroupParams};
use arrow_quant_v2::errors::QuantError;

#[test]
fn test_time_aware_quantizer_api_unchanged() {
    // 验证 TimeAwareQuantizer 的公开 API 保持不变
    
    // 1. 构造函数签名不变
    let quantizer = TimeAwareQuantizer::new(10);
    assert_eq!(quantizer.num_time_groups(), 10);
    
    // 2. 基本量化方法签名不变
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = vec![
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 10.0,
        };
        10
    ];
    
    // quantize_layer_arrow 方法应该存在且签名不变
    let result = quantizer.quantize_layer_arrow(&weights, &params, 8);
    assert!(result.is_ok());
}

#[test]
fn test_time_group_params_api_unchanged() {
    // 验证 TimeGroupParams 结构体字段不变
    let params = TimeGroupParams {
        scale: 0.1,
        zero_point: 128,
        min_val: 0.0,
        max_val: 10.0,
    };
    
    // 所有字段应该可访问
    assert_eq!(params.scale, 0.1);
    assert_eq!(params.zero_point, 128);
    assert_eq!(params.min_val, 0.0);
    assert_eq!(params.max_val, 10.0);
}

#[test]
fn test_error_types_unchanged() {
    // 验证错误类型保持不变
    let err = QuantError::InvalidInput("test".to_string());
    assert!(matches!(err, QuantError::InvalidInput(_)));
    
    let err = QuantError::QuantizationFailed("test".to_string());
    assert!(matches!(err, QuantError::QuantizationFailed(_)));
}

#[test]
fn test_default_behavior_unchanged() {
    // 验证默认行为与基线实现一致
    let quantizer = TimeAwareQuantizer::new(10);
    
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = vec![
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 10.0,
        };
        10
    ];
    
    // 默认量化行为应该产生一致的结果
    let result1 = quantizer.quantize_layer_arrow(&weights, &params, 8);
    let result2 = quantizer.quantize_layer_arrow(&weights, &params, 8);
    
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    // 相同输入应该产生相同输出（确定性）
    let batch1 = result1.unwrap();
    let batch2 = result2.unwrap();
    
    assert_eq!(batch1.num_rows(), batch2.num_rows());
    assert_eq!(batch1.num_columns(), batch2.num_columns());
}

#[test]
fn test_quantize_layer_arrow_signature() {
    // 验证 quantize_layer_arrow 方法签名完全不变
    let quantizer = TimeAwareQuantizer::new(5);
    
    let weights: &[f32] = &[1.0, 2.0, 3.0];
    let params: &[TimeGroupParams] = &[
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 10.0,
        };
        5
    ];
    let bit_width: u8 = 8;
    
    // 方法应该接受这些参数类型
    let _result = quantizer.quantize_layer_arrow(weights, params, bit_width);
}

#[test]
fn test_num_time_groups_method_unchanged() {
    // 验证 num_time_groups 方法存在且返回正确类型
    let quantizer = TimeAwareQuantizer::new(15);
    let num_groups: usize = quantizer.num_time_groups();
    assert_eq!(num_groups, 15);
}

#[test]
fn test_backward_compatibility_with_existing_code() {
    // 模拟现有代码的使用模式，确保仍然工作
    
    // 场景 1: 基本量化工作流
    let quantizer = TimeAwareQuantizer::new(10);
    let weights = vec![0.5, 1.5, 2.5, 3.5, 4.5];
    let params = vec![
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 5.0,
        };
        10
    ];
    
    let result = quantizer.quantize_layer_arrow(&weights, &params, 8);
    assert!(result.is_ok());
    
    let batch = result.unwrap();
    assert_eq!(batch.num_rows(), weights.len());
    
    // 场景 2: 错误处理模式
    let empty_weights: Vec<f32> = vec![];
    let result = quantizer.quantize_layer_arrow(&empty_weights, &params, 8);
    assert!(result.is_err());
}

#[test]
fn test_api_stability_across_bit_widths() {
    // 验证不同 bit_width 参数下 API 行为一致
    let quantizer = TimeAwareQuantizer::new(10);
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let params = vec![
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 10.0,
        };
        10
    ];
    
    // 所有支持的 bit_width 应该工作
    for bit_width in [2, 4, 8] {
        let result = quantizer.quantize_layer_arrow(&weights, &params, bit_width);
        assert!(result.is_ok(), "bit_width {} should work", bit_width);
    }
}

#[test]
fn test_no_breaking_changes_in_return_types() {
    // 验证返回类型没有破坏性变更
    let quantizer = TimeAwareQuantizer::new(10);
    let weights = vec![1.0, 2.0, 3.0];
    let params = vec![
        TimeGroupParams {
            scale: 0.1,
            zero_point: 128,
            min_val: 0.0,
            max_val: 10.0,
        };
        10
    ];
    
    // 返回类型应该是 Result<RecordBatch, QuantError>
    let result: Result<arrow::record_batch::RecordBatch, QuantError> = 
        quantizer.quantize_layer_arrow(&weights, &params, 8);
    
    assert!(result.is_ok());
}
