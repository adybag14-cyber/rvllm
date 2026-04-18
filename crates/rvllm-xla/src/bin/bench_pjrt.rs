#[cfg(not(feature = "tpu"))]
fn main() {
    eprintln!("ERROR: bench-pjrt requires the 'tpu' feature.");
    eprintln!("Build with: cargo run --release -p rvllm-xla --features tpu --bin bench-pjrt");
    std::process::exit(1);
}

#[cfg(feature = "tpu")]
fn main() {
    tpu_main::run();
}

#[cfg(feature = "tpu")]
mod tpu_main {
    use std::path::PathBuf;
    use std::time::Instant;

    use clap::Parser;
    use rvllm_xla::client::PjrtClientHandle;
    use rvllm_xla::ffi::PjrtElementType;
    use rvllm_xla::mlir_parser::{
        dtype_display_name, dtype_size_bytes, format_shape, parse_mlir_inputs,
    };

    #[derive(Parser)]
    #[command(name = "bench-pjrt", about = "Benchmark StableHLO kernels via PJRT")]
    struct Args {
        #[arg(long, default_value = "tpu/out/")]
        mlir_dir: PathBuf,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        #[arg(long)]
        only: Option<String>,
    }

    struct KernelResult {
        name: String,
        status: String,
        median_us: f64,
        first_input_shape: String,
        first_input_dtype: String,
    }

    pub fn run() {
        let args = Args::parse();
        let dir = &args.mlir_dir;

        if !dir.exists() || !dir.is_dir() {
            eprintln!("ERROR: MLIR directory '{}' does not exist", dir.display());
            std::process::exit(1);
        }

        let mut client = PjrtClientHandle::new().unwrap_or_else(|e| {
            eprintln!("ERROR: failed to create PJRT client: {e}");
            std::process::exit(1);
        });
        eprintln!("PJRT client ready, {} device(s)", client.num_devices());

        let opts_path = dir.join("compile_options.pb");
        if opts_path.exists() {
            let opts = std::fs::read(&opts_path).unwrap();
            eprintln!("loaded compile_options.pb ({} bytes)", opts.len());
            client.set_compile_options(opts);
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let ext = e.path().extension().and_then(|x| x.to_str()).map(|s| s.to_string());
                let stem = e.path().file_stem().and_then(|s| s.to_str()).map(|s| s.to_string());
                let is_mlir = ext.as_deref() == Some("mlir") || ext.as_deref() == Some("mlirbc");
                let not_dot = stem.as_deref().map(|s| !s.starts_with('.')).unwrap_or(false);
                is_mlir && not_dot
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());
        // deduplicate: if both .mlir and .mlirbc exist, prefer .mlirbc
        {
            let mut seen = std::collections::HashSet::new();
            let mut deduped = Vec::new();
            // sort so .mlirbc comes after .mlir, then keep the last (bytecode)
            entries.sort_by(|a, b| a.path().cmp(&b.path()));
            for e in entries {
                let stem = e.path().file_stem().unwrap().to_str().unwrap().to_string();
                seen.insert(stem.clone());
                deduped.retain(|d: &std::fs::DirEntry| {
                    d.path().file_stem().unwrap().to_str().unwrap() != stem
                });
                deduped.push(e);
            }
            entries = deduped;
            entries.sort_by_key(|e| e.file_name());
        }

        let mut results = Vec::new();

        for entry in &entries {
            let path = entry.path();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            if let Some(ref only) = args.only {
                if stem != only.as_str() {
                    continue;
                }
            }
            let is_bytecode = path.extension().and_then(|e| e.to_str()) == Some("mlirbc");

            // For shape parsing, need text MLIR (either from .mlir sibling or skip)
            let text_path = path.with_extension("mlir");
            let mlir_text = if is_bytecode && text_path.exists() {
                std::fs::read_to_string(&text_path).ok()
            } else if !is_bytecode {
                std::fs::read_to_string(&path).ok()
            } else {
                None
            };

            let code_bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    print_fail(stem, &format!("read error: {e}"));
                    results.push(KernelResult {
                        name: stem.to_string(),
                        status: "FAIL".into(),
                        median_us: 0.0,
                        first_input_shape: "".into(),
                        first_input_dtype: "".into(),
                    });
                    continue;
                }
            };

            let inputs = match mlir_text.as_deref().map(parse_mlir_inputs) {
                Some(Ok(i)) => i,
                Some(Err(e)) => {
                    print_fail(stem, &format!("parse error: {e}"));
                    results.push(KernelResult {
                        name: stem.to_string(),
                        status: "FAIL".into(),
                        median_us: 0.0,
                        first_input_shape: "".into(),
                        first_input_dtype: "".into(),
                    });
                    continue;
                }
                None => {
                    print_fail(stem, "no .mlir text for shape parsing");
                    results.push(KernelResult {
                        name: stem.to_string(),
                        status: "FAIL".into(),
                        median_us: 0.0,
                        first_input_shape: "".into(),
                        first_input_dtype: "".into(),
                    });
                    continue;
                }
            };

            let first_shape = if inputs.is_empty() {
                "()".to_string()
            } else {
                format_shape(&inputs[0].shape)
            };
            let first_dtype = if inputs.is_empty() {
                "unknown".to_string()
            } else {
                dtype_display_name(inputs[0].dtype).to_string()
            };

            eprintln!("  >> compiling {stem}...");
            let compiled = match if is_bytecode {
                client.compile_bytecode(&code_bytes)
            } else {
                client.compile(mlir_text.as_deref().unwrap_or(""))
            } {
                Ok(c) => c,
                Err(e) => {
                    print_fail(stem, &format!("compile error: {e}"));
                    results.push(KernelResult {
                        name: stem.to_string(),
                        status: "FAIL".into(),
                        median_us: 0.0,
                        first_input_shape: first_shape,
                        first_input_dtype: first_dtype,
                    });
                    continue;
                }
            };

            eprintln!("  >> compiled OK, creating {} buffers...", inputs.len());
            let buffers: Result<Vec<_>, _> = inputs
                .iter()
                .map(|inp| {
                    let num_elems: usize =
                        inp.shape.iter().map(|&d| d as usize).product::<usize>().max(1);
                    let elem_bytes = dtype_size_bytes(inp.dtype);
                    let total_bytes = num_elems * elem_bytes;
                    let data = make_random_bytes(total_bytes, inp.dtype);
                    client.buffer_from_host(&data, &inp.shape, inp.dtype, 0)
                })
                .collect();

            let buffers = match buffers {
                Ok(b) => b,
                Err(e) => {
                    print_fail(stem, &format!("buffer upload error: {e}"));
                    results.push(KernelResult {
                        name: stem.to_string(),
                        status: "FAIL".into(),
                        median_us: 0.0,
                        first_input_shape: first_shape,
                        first_input_dtype: first_dtype,
                    });
                    continue;
                }
            };

            let buf_refs: Vec<_> = buffers.iter().collect();

            eprintln!("  >> buffers OK, executing warmup...");
            // warmup
            let mut ok = true;
            for _ in 0..args.warmup {
                if let Err(e) = client.execute(&compiled, &buf_refs) {
                    print_fail(stem, &format!("exec error: {e}"));
                    ok = false;
                    break;
                }
            }
            if !ok {
                results.push(KernelResult {
                    name: stem.to_string(),
                    status: "FAIL".into(),
                    median_us: 0.0,
                    first_input_shape: first_shape,
                    first_input_dtype: first_dtype,
                });
                continue;
            }

            // timed runs
            let mut timings = Vec::with_capacity(args.iters);
            for _ in 0..args.iters {
                let t0 = Instant::now();
                let _ = client.execute(&compiled, &buf_refs);
                timings.push(t0.elapsed());
            }
            timings.sort();
            let median = timings[timings.len() / 2];
            let median_us = median.as_secs_f64() * 1_000_000.0;

            println!(
                "[OK            ] {:<28} OK   {:>8.1}us    {}:{}",
                stem, median_us, first_shape, first_dtype
            );

            results.push(KernelResult {
                name: stem.to_string(),
                status: "OK".into(),
                median_us,
                first_input_shape: first_shape,
                first_input_dtype: first_dtype,
            });
        }

        // summary JSON
        let json_entries: Vec<String> = results
            .iter()
            .map(|r| {
                format!(
                    "    {{\n      \"name\": \"{}\",\n      \"status\": \"{}\",\n      \"median_us\": {:.1},\n      \"shape\": \"{}\",\n      \"dtype\": \"{}\"\n    }}",
                    r.name, r.status, r.median_us, r.first_input_shape, r.first_input_dtype
                )
            })
            .collect();
        let json = format!("{{\n  \"kernels\": [\n{}\n  ]\n}}\n", json_entries.join(",\n"));
        let json_path = "/tmp/pjrt_bench.json";
        std::fs::write(json_path, &json).unwrap_or_else(|e| {
            eprintln!("WARNING: failed to write {json_path}: {e}");
        });
        eprintln!("Summary written to {json_path}");
    }

    fn print_fail(name: &str, msg: &str) {
        println!("[FAIL          ] {:<28} FAIL {}", name, msg);
    }

    fn make_random_bytes(total_bytes: usize, dtype: PjrtElementType) -> Vec<u8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut buf = vec![0u8; total_bytes];
        match dtype {
            PjrtElementType::F32 => {
                let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut buf);
                for v in floats.iter_mut() {
                    *v = rng.gen_range(-1.0f32..1.0f32);
                }
            }
            PjrtElementType::F16 | PjrtElementType::BF16 => {
                let halfs: &mut [u16] = bytemuck::cast_slice_mut(&mut buf);
                for v in halfs.iter_mut() {
                    let f: f32 = rng.gen_range(-1.0f32..1.0f32);
                    *v = half::f16::from_f32(f).to_bits();
                }
            }
            PjrtElementType::S32 => {
                let ints: &mut [i32] = bytemuck::cast_slice_mut(&mut buf);
                for v in ints.iter_mut() {
                    *v = rng.gen_range(0i32..128);
                }
            }
            PjrtElementType::S64 => {
                let ints: &mut [i64] = bytemuck::cast_slice_mut(&mut buf);
                for v in ints.iter_mut() {
                    *v = rng.gen_range(0i64..128);
                }
            }
            _ => {
                rng.fill(&mut buf[..]);
            }
        }
        buf
    }
}
