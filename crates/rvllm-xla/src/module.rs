use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

use crate::buffer::XlaBuffer;
use crate::Result;

pub struct LoadedExecutable {
    name: String,
    // Will hold *mut PJRT_LoadedExecutable once FFI lands.
    num_inputs: usize,
    num_outputs: usize,
}

impl LoadedExecutable {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn execute(&self, _inputs: &[&XlaBuffer]) -> Result<Vec<XlaBuffer>> {
        // Will call PJRT_LoadedExecutable_Execute.
        Err(crate::LLMError::GpuError(format!(
            "PJRT FFI not yet implemented -- cannot execute module '{}'",
            self.name
        )))
    }
}

pub struct ModuleLoader {
    modules: HashMap<String, LoadedExecutable>,
}

impl ModuleLoader {
    pub fn new(mlir_dir: &Path) -> Result<Self> {
        let mut loader = Self {
            modules: HashMap::new(),
        };

        if !mlir_dir.exists() || !mlir_dir.is_dir() {
            return Err(crate::LLMError::GpuError(format!(
                "MLIR directory '{}' does not exist",
                mlir_dir.display()
            )));
        }

        loader.load_directory(mlir_dir)?;
        Ok(loader)
    }

    pub fn empty() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    pub fn load_mlir(&mut self, name: &str, mlir_text: &str) -> Result<()> {
        // Will:
        // 1. Serialize mlir_text to StableHLO bytecode
        // 2. Call PJRT_Client_Compile
        // 3. Store the LoadedExecutable
        debug!(module = name, len = mlir_text.len(), "loading StableHLO module");

        let (num_inputs, num_outputs) = parse_mlir_signature(mlir_text)?;

        let exe = LoadedExecutable {
            name: name.to_string(),
            num_inputs,
            num_outputs,
        };

        self.modules.insert(name.to_string(), exe);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Result<&LoadedExecutable> {
        self.modules.get(name).ok_or_else(|| {
            crate::LLMError::GpuError(format!("module '{}' not loaded", name))
        })
    }

    pub fn require(&self, name: &str) -> &LoadedExecutable {
        self.get(name).unwrap_or_else(|e| {
            panic!(
                "REQUIRED StableHLO module missing: {} -- {}. \
                 Run `python3 tpu/harness/emit_all.py` to emit .mlir files.",
                name, e
            )
        })
    }

    pub fn has_module(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    pub fn loaded_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    fn load_directory(&mut self, dir: &Path) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::LLMError::GpuError(format!(
                "cannot read MLIR dir '{}': {e}",
                dir.display()
            ))
        })?;

        let mut count = 0u32;
        for entry in entries {
            let entry = entry.map_err(|e| {
                crate::LLMError::GpuError(format!("readdir error: {e}"))
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("mlir") {
                continue;
            }
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| {
                    crate::LLMError::GpuError(format!(
                        "invalid filename: {}",
                        path.display()
                    ))
                })?;

            let text = std::fs::read_to_string(&path).map_err(|e| {
                crate::LLMError::GpuError(format!(
                    "failed to read '{}': {e}",
                    path.display()
                ))
            })?;

            self.load_mlir(stem, &text)?;
            count += 1;
        }

        info!(dir = %dir.display(), count, "loaded StableHLO modules");
        Ok(())
    }
}

fn parse_mlir_signature(mlir: &str) -> Result<(usize, usize)> {
    // Quick parse of `func.func public @main(%arg0: ..., %arg1: ...) -> (...)`
    // to extract input/output counts. Full MLIR parsing is not needed here;
    // PJRT_Client_Compile will validate the program.
    let main_line = mlir
        .lines()
        .find(|l| l.contains("func.func") && l.contains("@main"))
        .ok_or_else(|| {
            crate::LLMError::GpuError(
                "MLIR module has no `func.func public @main` -- invalid StableHLO".into(),
            )
        })?;

    let num_inputs = main_line.matches("%arg").count();

    // Count outputs: look for `->` then count tensor types or tuple elements
    let num_outputs = if main_line.contains("->") {
        // Simple heuristic: count tensor< occurrences after ->
        let after_arrow = main_line.split("->").nth(1).unwrap_or("");
        after_arrow.matches("tensor<").count().max(1)
    } else {
        0
    };

    Ok((num_inputs, num_outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rms_norm_signature() {
        let mlir = r#"module @jit_rms_norm {
  func.func public @main(%arg0: tensor<128x4096xf32>, %arg1: tensor<4096xf32>) -> (tensor<128x4096xf32>) {
    return %0 : tensor<128x4096xf32>
  }
}"#;
        let (inputs, outputs) = parse_mlir_signature(mlir).unwrap();
        assert_eq!(inputs, 2);
        assert_eq!(outputs, 1);
    }

    #[test]
    fn parse_no_main_fails() {
        let mlir = "module @jit_foo { }";
        assert!(parse_mlir_signature(mlir).is_err());
    }

    #[test]
    fn empty_loader() {
        let loader = ModuleLoader::empty();
        assert!(loader.loaded_modules().is_empty());
        assert!(!loader.has_module("anything"));
    }
}
