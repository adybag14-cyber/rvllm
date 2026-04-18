use crate::ffi::PjrtElementType;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorSig {
    pub shape: Vec<i64>,
    pub dtype: PjrtElementType,
}

pub fn parse_mlir_inputs(mlir: &str) -> Result<Vec<TensorSig>, String> {
    let main_line = find_main_line(mlir)?;
    let paren_start = main_line.find('(').ok_or("no '(' in func signature")?;
    let arrow_pos = main_line.find("->").unwrap_or(main_line.len());
    let args_region = &main_line[paren_start..arrow_pos];

    let mut inputs = Vec::new();
    for cap in args_region.split("%arg") {
        if let Some(sig) = parse_tensor_from_fragment(cap) {
            inputs.push(sig);
        }
    }
    Ok(inputs)
}

pub fn parse_mlir_outputs(mlir: &str) -> Result<Vec<TensorSig>, String> {
    let main_line = find_main_line(mlir)?;
    let arrow_pos = match main_line.find("->") {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    let after_arrow = &main_line[arrow_pos + 2..];

    let mut outputs = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let bytes = after_arrow.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let chunk = &after_arrow[start..=i];
                    for part in chunk.split("tensor<") {
                        if let Some(sig) = parse_tensor_type_tail(part) {
                            outputs.push(sig);
                        }
                    }
                    break;
                }
            }
            _ => {}
        }
        if depth == 1 && i == 0 {
            start = 1;
        }
    }
    if outputs.is_empty() {
        for part in after_arrow.split("tensor<") {
            if let Some(sig) = parse_tensor_type_tail(part) {
                outputs.push(sig);
            }
        }
    }
    Ok(outputs)
}

fn find_main_line(mlir: &str) -> Result<String, String> {
    let mut combined = String::new();
    let mut in_func = false;
    for line in mlir.lines() {
        let trimmed = line.trim();
        if trimmed.contains("func.func") && trimmed.contains("@main") {
            in_func = true;
        }
        if in_func {
            combined.push_str(trimmed);
            combined.push(' ');
            if trimmed.contains('{') {
                break;
            }
        }
    }
    if combined.is_empty() {
        return Err("no func.func @main found".into());
    }
    Ok(combined)
}

fn parse_tensor_from_fragment(frag: &str) -> Option<TensorSig> {
    let idx = frag.find("tensor<")?;
    let tail = &frag[idx + 7..];
    parse_tensor_type_tail(tail)
}

fn parse_tensor_type_tail(tail: &str) -> Option<TensorSig> {
    let end = tail.find('>')?;
    let spec = &tail[..end];
    parse_tensor_spec(spec)
}

fn parse_tensor_spec(spec: &str) -> Option<TensorSig> {
    // spec like "128x4096xf32" or "4096xf32" or "f32" (scalar tensor)
    let dtype_str = spec.rsplit('x').next()?;
    let dtype = parse_element_type(dtype_str)?;
    let mut shape = Vec::new();
    let parts: Vec<&str> = spec.split('x').collect();
    if parts.len() > 1 {
        for dim_str in &parts[..parts.len() - 1] {
            let d: i64 = dim_str.parse().ok()?;
            shape.push(d);
        }
    }
    Some(TensorSig { shape, dtype })
}

fn parse_element_type(s: &str) -> Option<PjrtElementType> {
    match s {
        "f32" => Some(PjrtElementType::F32),
        "f16" => Some(PjrtElementType::F16),
        "bf16" => Some(PjrtElementType::BF16),
        "f64" => Some(PjrtElementType::F64),
        "i32" | "si32" => Some(PjrtElementType::S32),
        "i64" | "si64" => Some(PjrtElementType::S64),
        "i16" | "si16" => Some(PjrtElementType::S16),
        "i8" | "si8" => Some(PjrtElementType::S8),
        "ui32" => Some(PjrtElementType::U32),
        "ui16" => Some(PjrtElementType::U16),
        "ui8" => Some(PjrtElementType::U8),
        "i1" => Some(PjrtElementType::PRED),
        _ => None,
    }
}

pub fn dtype_size_bytes(dtype: PjrtElementType) -> usize {
    match dtype {
        PjrtElementType::F64 | PjrtElementType::S64 | PjrtElementType::U64 | PjrtElementType::C64 => 8,
        PjrtElementType::F32 | PjrtElementType::S32 | PjrtElementType::U32 => 4,
        PjrtElementType::F16 | PjrtElementType::BF16 | PjrtElementType::S16 | PjrtElementType::U16 => 2,
        PjrtElementType::S8 | PjrtElementType::U8 | PjrtElementType::PRED
        | PjrtElementType::F8E5M2 | PjrtElementType::F8E4M3FN => 1,
        PjrtElementType::C128 => 16,
        PjrtElementType::INVALID => 0,
    }
}

pub fn dtype_display_name(dtype: PjrtElementType) -> &'static str {
    match dtype {
        PjrtElementType::F32 => "float32",
        PjrtElementType::F16 => "float16",
        PjrtElementType::BF16 => "bfloat16",
        PjrtElementType::F64 => "float64",
        PjrtElementType::S32 => "int32",
        PjrtElementType::S64 => "int64",
        PjrtElementType::S16 => "int16",
        PjrtElementType::S8 => "int8",
        PjrtElementType::U32 => "uint32",
        PjrtElementType::U16 => "uint16",
        PjrtElementType::U8 => "uint8",
        PjrtElementType::PRED => "bool",
        _ => "unknown",
    }
}

pub fn format_shape(shape: &[i64]) -> String {
    if shape.is_empty() {
        return "()".to_string();
    }
    let inner: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("({})", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rms_norm() {
        let mlir = r#"module @jit_rms_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x4096xf32>, %arg1: tensor<4096xf32>) -> (tensor<128x4096xf32> {jax.result_info = "result"}) {
    return %0 : tensor<128x4096xf32>
  }
}"#;
        let inputs = parse_mlir_inputs(mlir).unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].shape, vec![128, 4096]);
        assert_eq!(inputs[0].dtype, PjrtElementType::F32);
        assert_eq!(inputs[1].shape, vec![4096]);
        assert_eq!(inputs[1].dtype, PjrtElementType::F32);

        let outputs = parse_mlir_outputs(mlir).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, vec![128, 4096]);
        assert_eq!(outputs[0].dtype, PjrtElementType::F32);
    }

    #[test]
    fn parse_bf16_gemm() {
        let mlir = r#"module @jit_gemm_bf16 attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<128x4096xbf16>, %arg1: tensor<4096x4096xbf16>) -> (tensor<128x4096xbf16> {jax.result_info = "result"}) {
    return %0 : tensor<128x4096xbf16>
  }
}"#;
        let inputs = parse_mlir_inputs(mlir).unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].dtype, PjrtElementType::BF16);
        assert_eq!(inputs[1].shape, vec![4096, 4096]);
    }

    #[test]
    fn parse_multi_output() {
        let mlir = r#"module @jit_fused {
  func.func public @main(%arg0: tensor<128x4096xf32>, %arg1: tensor<128x4096xf32>, %arg2: tensor<4096xf32>) -> (tensor<128x4096xf32> {jax.result_info = "result[0]"}, tensor<128x4096xf32> {jax.result_info = "result[1]"}) {
    return %0, %1 : tensor<128x4096xf32>, tensor<128x4096xf32>
  }
}"#;
        let outputs = parse_mlir_outputs(mlir).unwrap();
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn parse_i32_output() {
        let mlir = r#"module @jit_argmax {
  func.func public @main(%arg0: tensor<128x32000xf32>) -> (tensor<128xi32> {jax.result_info = "result"}) {
    return %0 : tensor<128xi32>
  }
}"#;
        let inputs = parse_mlir_inputs(mlir).unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].dtype, PjrtElementType::F32);
        let outputs = parse_mlir_outputs(mlir).unwrap();
        assert_eq!(outputs[0].dtype, PjrtElementType::S32);
        assert_eq!(outputs[0].shape, vec![128]);
    }

    #[test]
    fn parse_f16_tensors() {
        let mlir = r#"module @jit_reshape {
  func.func public @main(%arg0: tensor<128x8x128xf16>, %arg1: tensor<128x8x128xf16>, %arg2: tensor<1024x16x8x128xf16>, %arg3: tensor<1024x16x8x128xf16>, %arg4: tensor<128xi32>) -> (tensor<1024x16x8x128xf16> {jax.result_info = "result[0]"}, tensor<1024x16x8x128xf16> {jax.result_info = "result[1]"}) {
    return %0, %1 : tensor<1024x16x8x128xf16>, tensor<1024x16x8x128xf16>
  }
}"#;
        let inputs = parse_mlir_inputs(mlir).unwrap();
        assert_eq!(inputs.len(), 5);
        assert_eq!(inputs[0].dtype, PjrtElementType::F16);
        assert_eq!(inputs[4].dtype, PjrtElementType::S32);
    }

    #[test]
    fn parse_paged_attention() {
        let mlir = r#"module @jit_paged_attention {
  func.func public @main(%arg0: tensor<8x32x128xbf16>, %arg1: tensor<1024x16x8x128xbf16>, %arg2: tensor<1024x16x8x128xbf16>, %arg3: tensor<8x256xi32>, %arg4: tensor<8xi32>) -> (tensor<8x32x128xbf16> {jax.result_info = "result"}) {
    return %0 : tensor<8x32x128xbf16>
  }
}"#;
        let inputs = parse_mlir_inputs(mlir).unwrap();
        assert_eq!(inputs.len(), 5);
        assert_eq!(inputs[0].shape, vec![8, 32, 128]);
        assert_eq!(inputs[3].shape, vec![8, 256]);
        assert_eq!(inputs[3].dtype, PjrtElementType::S32);
        assert_eq!(inputs[4].shape, vec![8]);
    }

    #[test]
    fn parse_real_mlir_files() {
        let mlir_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tpu/out");
        if !mlir_dir.exists() {
            return; // skip if not present
        }
        for entry in std::fs::read_dir(&mlir_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("mlir") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let stem = path.file_stem().unwrap().to_str().unwrap();
            let inputs = parse_mlir_inputs(&text)
                .unwrap_or_else(|e| panic!("{stem}: input parse failed: {e}"));
            assert!(!inputs.is_empty(), "{stem}: expected at least one input");
            let outputs = parse_mlir_outputs(&text)
                .unwrap_or_else(|e| panic!("{stem}: output parse failed: {e}"));
            assert!(!outputs.is_empty(), "{stem}: expected at least one output");
            for (i, inp) in inputs.iter().enumerate() {
                assert!(
                    dtype_size_bytes(inp.dtype) > 0,
                    "{stem} input {i}: invalid dtype {:?}",
                    inp.dtype
                );
            }
        }
    }
}
