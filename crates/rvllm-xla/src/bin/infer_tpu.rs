#[cfg(not(feature = "tpu"))]
fn main() {
    eprintln!("ERROR: infer-tpu requires the 'tpu' feature.");
    eprintln!("Build with: cargo run --release -p rvllm-xla --features tpu --bin infer-tpu");
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

    #[derive(Parser)]
    #[command(name = "infer-tpu", about = "Run rvLLM inference on TPU via PJRT")]
    struct Args {
        #[arg(long, default_value = "tpu/out/")]
        mlir_dir: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value = "Hello, world")]
        prompt: String,
    }

    // Hardcoded for the MLIR modules we have (Llama 3.1 8B shapes)
    const HIDDEN: usize = 4096;
    const NUM_HEADS: usize = 32;
    const NUM_KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const INTERMEDIATE: usize = 14336;
    const NUM_LAYERS: usize = 32;
    const VOCAB: usize = 128256;
    const BLOCK_SIZE: usize = 16;
    const NUM_BLOCKS: usize = 1024;
    const MAX_BLOCKS_PER_SEQ: usize = 256;
    const BATCH: usize = 8; // padded batch from layer_decode MLIR
    const EMBED_BATCH: usize = 128; // padded batch from embedding MLIR

    pub fn run() {
        let args = Args::parse();

        // Load compile options
        let opts_path = args.mlir_dir.join("compile_options.pb");
        let compile_opts = if opts_path.exists() {
            Some(std::fs::read(&opts_path).unwrap())
        } else {
            None
        };

        let mut client = PjrtClientHandle::new().unwrap_or_else(|e| {
            eprintln!("FATAL: PJRT init failed: {e}");
            std::process::exit(1);
        });
        eprintln!("PJRT client: {} device(s)", client.num_devices());

        if let Some(opts) = compile_opts {
            client.set_compile_options(opts);
        }

        // Compile the modules we need
        eprintln!("compiling modules...");
        let t0 = Instant::now();

        let embed_mlir = std::fs::read_to_string(args.mlir_dir.join("embedding_gather.mlir"))
            .expect("embedding_gather.mlir");
        let layer_mlir = std::fs::read_to_string(args.mlir_dir.join("persistent_layer_decode.mlir"))
            .expect("persistent_layer_decode.mlir");
        let head_mlir = std::fs::read_to_string(args.mlir_dir.join("fused_lm_head_argmax.mlir"))
            .expect("fused_lm_head_argmax.mlir");
        let norm_mlir = std::fs::read_to_string(args.mlir_dir.join("fused_residual_rmsnorm.mlir"))
            .expect("fused_residual_rmsnorm.mlir");

        let embed_exe = client.compile(&embed_mlir).expect("compile embedding");
        let layer_exe = client.compile(&layer_mlir).expect("compile layer");
        let head_exe = client.compile(&head_mlir).expect("compile lm_head");
        let norm_exe = client.compile(&norm_mlir).expect("compile norm");
        eprintln!("compiled 4 modules in {:.1}s", t0.elapsed().as_secs_f32());

        // Load model weights from safetensors
        eprintln!("loading weights from {:?}...", args.model_dir);
        let t0 = Instant::now();
        let weights = load_weights(&client, &args.model_dir);
        eprintln!("loaded {} layer weight sets + embed/head in {:.1}s",
            weights.layers.len(), t0.elapsed().as_secs_f32());

        // Tokenize prompt (simple: just use bytes as token IDs for now,
        // real tokenizer would go here)
        let prompt_ids: Vec<i32> = args.prompt.bytes().map(|b| b as i32).collect();
        eprintln!("prompt: {:?} ({} tokens)", args.prompt, prompt_ids.len());

        // Initialize KV caches (zeros)
        let kv_bytes = NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * 2; // bf16
        let kv_shape: Vec<i64> = vec![NUM_BLOCKS as i64, BLOCK_SIZE as i64,
            NUM_KV_HEADS as i64, HEAD_DIM as i64];
        let zero_kv = vec![0u8; kv_bytes];

        let mut k_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| client.buffer_from_host(&zero_kv, &kv_shape, PjrtElementType::BF16, 0).unwrap())
            .collect();
        let mut v_caches: Vec<_> = (0..NUM_LAYERS)
            .map(|_| client.buffer_from_host(&zero_kv, &kv_shape, PjrtElementType::BF16, 0).unwrap())
            .collect();

        // Block tables: each seq gets sequential blocks
        let mut bt_host = vec![0i32; BATCH * MAX_BLOCKS_PER_SEQ];
        for seq in 0..BATCH {
            for b in 0..MAX_BLOCKS_PER_SEQ {
                bt_host[seq * MAX_BLOCKS_PER_SEQ + b] = (seq * MAX_BLOCKS_PER_SEQ + b) as i32;
            }
        }
        let bt_buf = client.buffer_from_host(
            bytemuck::cast_slice(&bt_host),
            &[BATCH as i64, MAX_BLOCKS_PER_SEQ as i64],
            PjrtElementType::S32, 0,
        ).unwrap();

        let mut generated = Vec::new();
        let mut context_len: i32 = 0;
        let mut ttft_ns: Option<u128> = None;
        let mut decode_start: Option<Instant> = None;

        // Process prompt tokens one at a time (decode mode)
        eprintln!("--- inference ---");
        let prompt_start = Instant::now();
        let total_steps = prompt_ids.len() + args.max_tokens;
        for step in 0..total_steps {
            let token_id = if step < prompt_ids.len() {
                prompt_ids[step]
            } else {
                *generated.last().unwrap_or(&0)
            };

            // Embedding uses EMBED_BATCH=128, layer uses BATCH=8
            let mut tok_batch = vec![0i32; EMBED_BATCH];
            tok_batch[0] = token_id;

            let tok_buf = client.buffer_from_host(
                bytemuck::cast_slice(&tok_batch),
                &[EMBED_BATCH as i64],
                PjrtElementType::S32, 0,
            ).unwrap();

            // Embedding gather -> [EMBED_BATCH, HIDDEN] f32
            let embed_out = client.execute(&embed_exe, &[&weights.embedding, &tok_buf]).unwrap();
            let hidden_f32 = &embed_out[0];

            // DtoH full embed output, convert to bf16, take first BATCH rows
            let embed_f32_bytes = EMBED_BATCH * HIDDEN * 4;
            let mut hidden_bytes = vec![0u8; embed_f32_bytes];
            client.buffer_to_host(hidden_f32, &mut hidden_bytes).unwrap();
            // Take only first BATCH rows
            let batch_f32_bytes = &hidden_bytes[..BATCH * HIDDEN * 4];
            let hidden_bf16 = f32_bytes_to_bf16(batch_f32_bytes);
            let hidden_buf = client.buffer_from_host(
                &hidden_bf16,
                &[BATCH as i64, HIDDEN as i64],
                PjrtElementType::BF16, 0,
            ).unwrap();

            // Residual starts as hidden
            let mut residual_buf = client.buffer_from_host(
                &vec![0u8; BATCH * HIDDEN * 2],
                &[BATCH as i64, HIDDEN as i64],
                PjrtElementType::BF16, 0,
            ).unwrap();
            let mut current_hidden = hidden_buf;

            // Positions and slot mapping
            let pos_host: Vec<i32> = (0..BATCH).map(|_| context_len).collect();
            let slot_host: Vec<i32> = (0..BATCH).map(|i| {
                context_len * (BATCH as i32) + i as i32
            }).collect();
            let ctx_host: Vec<i32> = (0..BATCH).map(|_| context_len + 1).collect();

            let pos_buf = client.buffer_from_host(
                bytemuck::cast_slice(&pos_host), &[BATCH as i64],
                PjrtElementType::S32, 0,
            ).unwrap();
            let slot_buf = client.buffer_from_host(
                bytemuck::cast_slice(&slot_host), &[BATCH as i64],
                PjrtElementType::S32, 0,
            ).unwrap();
            let ctx_buf = client.buffer_from_host(
                bytemuck::cast_slice(&ctx_host), &[BATCH as i64],
                PjrtElementType::S32, 0,
            ).unwrap();

            // Run all layers
            for layer in 0..NUM_LAYERS {
                let layer_out = client.execute(&layer_exe, &[
                    &current_hidden,  // arg0: hidden [8, 4096] bf16
                    &residual_buf,    // arg1: residual [8, 4096] bf16
                    &weights.layers[layer].input_norm, // arg2: norm gamma [4096] bf16
                    &weights.layers[layer].qkv,        // arg3: QKV [4096, 6144] bf16
                    &weights.layers[layer].o_proj,      // arg4: O [4096, 4096] bf16
                    &weights.layers[layer].mlp_norm,    // arg5: mlp norm [4096] bf16
                    &weights.layers[layer].gate_up,     // arg6: gate_up [4096, 28672] bf16
                    &weights.layers[layer].down,        // arg7: down [14336, 4096] bf16
                    &weights.rope_cos,                  // arg8: cos [4096, 64] f32
                    &weights.rope_sin,                  // arg9: sin [4096, 64] f32
                    &pos_buf,                           // arg10: positions [8] i32
                    &slot_buf,                          // arg11: slot_mapping [8] i32
                    &k_caches[layer],                   // arg12: k_cache
                    &v_caches[layer],                   // arg13: v_cache
                    &bt_buf,                            // arg14: block_tables [8, 256] i32
                    &ctx_buf,                           // arg15: context_lens [8] i32
                ]).unwrap();

                // Outputs: [hidden, residual, k_cache, v_cache]
                let mut outs = layer_out.into_iter();
                current_hidden = outs.next().unwrap();
                residual_buf = outs.next().unwrap();
                k_caches[layer] = outs.next().unwrap();
                v_caches[layer] = outs.next().unwrap();
            }

            // Final norm + LM head + argmax
            // lm_head module expects [EMBED_BATCH, HIDDEN] f16.
            // Read bf16 hidden, convert to f16, pad to EMBED_BATCH.
            let mut hid_bytes = vec![0u8; BATCH * HIDDEN * 2]; // bf16
            client.buffer_to_host(&current_hidden, &mut hid_bytes).unwrap();
            let hid_f16 = bf16_bytes_to_f16(&hid_bytes);
            // Pad to EMBED_BATCH
            let mut padded_f16 = vec![0u8; EMBED_BATCH * HIDDEN * 2];
            padded_f16[..hid_f16.len()].copy_from_slice(&hid_f16);
            let hid_f16_buf = client.buffer_from_host(
                &padded_f16,
                &[EMBED_BATCH as i64, HIDDEN as i64],
                PjrtElementType::F16, 0,
            ).unwrap();

            let head_out = client.execute(&head_exe, &[
                &hid_f16_buf,
                &weights.lm_head_f16,
            ]).unwrap();

            // Read sampled token (EMBED_BATCH tokens, take first)
            let mut token_bytes = vec![0u8; EMBED_BATCH * 4];
            client.buffer_to_host(&head_out[0], &mut token_bytes).unwrap();
            let tokens: &[i32] = bytemuck::cast_slice(&token_bytes);
            let sampled = tokens[0];

            // Debug: on first decode step, dump hidden and sampled info
            if step == prompt_ids.len() {
                eprintln!("[DEBUG] first decode: sampled={} tokens[0..4]={:?}",
                    sampled, &tokens[..4.min(tokens.len())]);
                // Read hidden state to check if nonzero
                let mut dbg_bytes = vec![0u8; BATCH * HIDDEN * 2];
                client.buffer_to_host(&current_hidden, &mut dbg_bytes).unwrap();
                let first8: Vec<f32> = (0..8).map(|i| {
                    let bits = u16::from_le_bytes([dbg_bytes[2*i], dbg_bytes[2*i+1]]);
                    half::bf16::from_bits(bits).to_f32()
                }).collect();
                eprintln!("[DEBUG] hidden[0..8] = {:?}", first8);
            }

            context_len += 1;

            if step < prompt_ids.len() {
                eprint!(".");
                if step == prompt_ids.len() - 1 {
                    use std::io::Write;
                    std::io::stderr().flush().ok();
                    ttft_ns = Some(prompt_start.elapsed().as_nanos());
                    decode_start = Some(Instant::now());
                    eprintln!("\nTTFT: {:.2}ms ({} prompt tokens)",
                        ttft_ns.unwrap() as f64 / 1_000_000.0, prompt_ids.len());
                    std::io::stderr().flush().ok();
                }
            } else {
                generated.push(sampled);
                if sampled == 1 || sampled == 2 || sampled == 128001 { // EOS
                    eprintln!("[EOS at step {}]", step);
                    break;
                }
                eprint!("[{}]", sampled);
            }
        }

        let total_elapsed = prompt_start.elapsed();
        let decode_elapsed = decode_start.map(|s| s.elapsed());

        eprintln!();
        eprintln!("=== Results ===");
        eprintln!("prompt tokens:    {}", prompt_ids.len());
        eprintln!("generated tokens: {}", generated.len());
        if let Some(ttft) = ttft_ns {
            eprintln!("TTFT:             {:.2}ms", ttft as f64 / 1_000_000.0);
        }
        if let Some(dt) = decode_elapsed {
            let toks = generated.len();
            if toks > 0 {
                let tok_s = toks as f64 / dt.as_secs_f64();
                let ms_per_tok = dt.as_secs_f64() * 1000.0 / toks as f64;
                eprintln!("decode tok/s:     {:.1}", tok_s);
                eprintln!("ms/token:         {:.1}", ms_per_tok);
            }
        }
        eprintln!("total time:       {:.2}s", total_elapsed.as_secs_f64());
        eprintln!("generated:        {:?}", &generated[..generated.len().min(20)]);
    }

    struct LayerWeights {
        input_norm: rvllm_xla::client::PjrtBufferHandle,
        qkv: rvllm_xla::client::PjrtBufferHandle,
        o_proj: rvllm_xla::client::PjrtBufferHandle,
        mlp_norm: rvllm_xla::client::PjrtBufferHandle,
        gate_up: rvllm_xla::client::PjrtBufferHandle,
        down: rvllm_xla::client::PjrtBufferHandle,
    }

    struct ModelWeights {
        embedding: rvllm_xla::client::PjrtBufferHandle,
        lm_head_f16: rvllm_xla::client::PjrtBufferHandle,
        rope_cos: rvllm_xla::client::PjrtBufferHandle,
        rope_sin: rvllm_xla::client::PjrtBufferHandle,
        layers: Vec<LayerWeights>,
    }

    fn load_weights(client: &PjrtClientHandle, model_dir: &PathBuf) -> ModelWeights {
        use std::collections::BTreeMap;

        let idx_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        // Find all safetensors files
        let shard_paths: Vec<PathBuf> = if idx_path.exists() {
            let idx: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(&idx_path).unwrap()
            ).unwrap();
            let wm = idx["weight_map"].as_object().unwrap();
            let mut shards: Vec<String> = wm.values()
                .map(|v| v.as_str().unwrap().to_string())
                .collect::<std::collections::HashSet<_>>()
                .into_iter().collect();
            shards.sort();
            shards.iter().map(|s| model_dir.join(s)).collect()
        } else if single_path.exists() {
            vec![single_path]
        } else {
            panic!("no safetensors found in {:?}", model_dir);
        };

        // Build tensor index
        let mut tensor_data: BTreeMap<String, (Vec<usize>, Vec<u8>, String)> = BTreeMap::new();
        for sp in &shard_paths {
            let mmap = unsafe { memmap2::Mmap::map(&std::fs::File::open(sp).unwrap()).unwrap() };
            let header_len = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&mmap[8..8+header_len]).unwrap();
            let data_start = 8 + header_len;
            for (name, info) in header.as_object().unwrap() {
                if name == "__metadata__" { continue; }
                let shape: Vec<usize> = info["shape"].as_array().unwrap()
                    .iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let dtype = info["dtype"].as_str().unwrap().to_string();
                let offsets = info["data_offsets"].as_array().unwrap();
                let start = offsets[0].as_u64().unwrap() as usize;
                let end = offsets[1].as_u64().unwrap() as usize;
                let bytes = mmap[data_start + start..data_start + end].to_vec();
                tensor_data.insert(name.clone(), (shape, bytes, dtype));
            }
        }

        eprintln!("  {} tensors indexed", tensor_data.len());

        let upload_bf16 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let bf16_bytes = match dtype.as_str() {
                "BF16" | "bf16" => bytes.clone(),
                "F16" | "f16" => f16_bytes_to_bf16(bytes),
                "F32" | "f32" | "F32" => f32_bytes_to_bf16(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&bf16_bytes, shape, PjrtElementType::BF16, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        let upload_f32 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let f32_bytes = match dtype.as_str() {
                "F32" | "f32" => bytes.clone(),
                "BF16" | "bf16" => bf16_bytes_to_f32(bytes),
                "F16" | "f16" => f16_bytes_to_f32(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&f32_bytes, shape, PjrtElementType::F32, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        let upload_f16 = |name: &str, shape: &[i64]| -> rvllm_xla::client::PjrtBufferHandle {
            let (_, bytes, dtype) = tensor_data.get(name)
                .unwrap_or_else(|| panic!("missing tensor: {name}"));
            let f16_bytes = match dtype.as_str() {
                "F16" | "f16" => bytes.clone(),
                "BF16" | "bf16" => bf16_bytes_to_f16(bytes),
                "F32" | "f32" => f32_bytes_to_f16(bytes),
                other => panic!("unsupported dtype {other} for {name}"),
            };
            client.buffer_from_host(&f16_bytes, shape, PjrtElementType::F16, 0)
                .unwrap_or_else(|e| panic!("upload {name}: {e}"))
        };

        // Embedding [vocab, hidden] f32 (MLIR module expects f32)
        let embedding = upload_f32("model.embed_tokens.weight",
            &[VOCAB as i64, HIDDEN as i64]);

        // LM head [vocab, hidden] f16
        let lm_head_f16 = if tensor_data.contains_key("lm_head.weight") {
            upload_f16("lm_head.weight", &[VOCAB as i64, HIDDEN as i64])
        } else {
            upload_f16("model.embed_tokens.weight", &[VOCAB as i64, HIDDEN as i64])
        };

        // RoPE cos/sin [max_pos, head_dim/2] f32
        let rope_cos = precompute_rope_cos(client, 10000.0);
        let rope_sin = precompute_rope_sin(client, 10000.0);

        // Per-layer weights
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for l in 0..NUM_LAYERS {
            let ln = |s: &str| format!("model.layers.{l}.{s}");

            let input_norm = upload_bf16(&ln("input_layernorm.weight"),
                &[HIDDEN as i64]);

            // Concat Q + K + V, transpose to [hidden, q_dim+2*kv_dim]
            // Safetensors: q=[q_dim, hidden], k=[kv_dim, hidden], v=[kv_dim, hidden]
            // MLIR expects: [hidden, q_dim+kv_dim+kv_dim] (contracting on dim 0 of weight = hidden)
            let q_name = ln("self_attn.q_proj.weight");
            let k_name = ln("self_attn.k_proj.weight");
            let v_name = ln("self_attn.v_proj.weight");
            let qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM;
            let qkv_bytes = concat_and_transpose_bf16(
                &tensor_data, &[&q_name, &k_name, &v_name], HIDDEN,
            );
            let qkv = client.buffer_from_host(&qkv_bytes,
                &[HIDDEN as i64, qkv_dim as i64],
                PjrtElementType::BF16, 0).unwrap();

            // O proj: safetensors [hidden, q_dim] -> MLIR [hidden, hidden]
            // dot_general contracts input[1] x weight[0], so weight is [q_dim, hidden]
            // safetensors stores [hidden, q_dim] -- need transpose
            let o_name = ln("self_attn.o_proj.weight");
            let o_bytes = transpose_bf16(&tensor_data, &o_name);
            let o_proj = client.buffer_from_host(&o_bytes,
                &[HIDDEN as i64, HIDDEN as i64],
                PjrtElementType::BF16, 0).unwrap();

            let mlp_norm = upload_bf16(&ln("post_attention_layernorm.weight"),
                &[HIDDEN as i64]);

            // gate+up: safetensors each [inter, hidden] -> MLIR [hidden, 2*inter]
            let g_name = ln("mlp.gate_proj.weight");
            let u_name = ln("mlp.up_proj.weight");
            let gate_up_bytes = concat_and_transpose_bf16(
                &tensor_data, &[&g_name, &u_name], HIDDEN,
            );
            let gate_up = client.buffer_from_host(&gate_up_bytes,
                &[HIDDEN as i64, (2 * INTERMEDIATE) as i64],
                PjrtElementType::BF16, 0).unwrap();

            // down: safetensors [hidden, inter] -> MLIR [inter, hidden]
            let d_name = ln("mlp.down_proj.weight");
            let down_bytes = transpose_bf16(&tensor_data, &d_name);
            let down = client.buffer_from_host(&down_bytes,
                &[INTERMEDIATE as i64, HIDDEN as i64],
                PjrtElementType::BF16, 0).unwrap();

            if l == 0 || l == NUM_LAYERS - 1 {
                eprintln!("  layer {l} loaded");
            } else if l == 1 {
                eprintln!("  ...");
            }

            layers.push(LayerWeights {
                input_norm,
                qkv,
                o_proj,
                mlp_norm,
                gate_up,
                down,
            });
        }

        ModelWeights {
            embedding,
            lm_head_f16,
            rope_cos,
            rope_sin,
            layers,
        }
    }

    fn get_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> (Vec<usize>, Vec<u8>) {
        let (shape, bytes, dtype) = tensors.get(name)
            .unwrap_or_else(|| panic!("missing tensor: {name}"));
        let bf16 = match dtype.as_str() {
            "BF16" | "bf16" => bytes.clone(),
            "F16" | "f16" => f16_bytes_to_bf16(bytes),
            "F32" | "f32" => f32_bytes_to_bf16(bytes),
            other => panic!("unsupported dtype {other} for {name}"),
        };
        (shape.clone(), bf16)
    }

    fn transpose_2d_bf16(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = (r * cols + c) * 2;
                let dst = (c * rows + r) * 2;
                out[dst] = data[src];
                out[dst + 1] = data[src + 1];
            }
        }
        out
    }

    fn transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        name: &str,
    ) -> Vec<u8> {
        let (shape, bf16) = get_bf16(tensors, name);
        assert_eq!(shape.len(), 2, "transpose requires 2D tensor: {name}");
        transpose_2d_bf16(&bf16, shape[0], shape[1])
    }

    fn concat_and_transpose_bf16(
        tensors: &BTreeMap<String, (Vec<usize>, Vec<u8>, String)>,
        names: &[&str],
        inner_dim: usize,
    ) -> Vec<u8> {
        // Each tensor is [out_dim, inner_dim]. Concat along out_dim, then
        // transpose to [inner_dim, total_out_dim].
        let mut total_out = 0usize;
        let mut parts = Vec::new();
        for name in names {
            let (shape, bf16) = get_bf16(tensors, name);
            assert_eq!(shape.len(), 2);
            assert_eq!(shape[1], inner_dim, "inner dim mismatch for {name}");
            total_out += shape[0];
            parts.push((shape[0], bf16));
        }
        // Concat [total_out, inner_dim]
        let mut cat = Vec::with_capacity(total_out * inner_dim * 2);
        for (_, data) in &parts {
            cat.extend_from_slice(data);
        }
        transpose_2d_bf16(&cat, total_out, inner_dim)
    }

    fn precompute_rope_cos(client: &PjrtClientHandle, theta: f32) -> rvllm_xla::client::PjrtBufferHandle {
        let half = HEAD_DIM / 2;
        let max_pos = 4096;
        let mut data = vec![0f32; max_pos * half];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / HEAD_DIM as f32);
                data[pos * half + i] = (pos as f32 * freq).cos();
            }
        }
        client.buffer_from_host(
            bytemuck::cast_slice(&data),
            &[max_pos as i64, half as i64],
            PjrtElementType::F32, 0,
        ).unwrap()
    }

    fn precompute_rope_sin(client: &PjrtClientHandle, theta: f32) -> rvllm_xla::client::PjrtBufferHandle {
        let half = HEAD_DIM / 2;
        let max_pos = 4096;
        let mut data = vec![0f32; max_pos * half];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / HEAD_DIM as f32);
                data[pos * half + i] = (pos as f32 * freq).sin();
            }
        }
        client.buffer_from_host(
            bytemuck::cast_slice(&data),
            &[max_pos as i64, half as i64],
            PjrtElementType::F32, 0,
        ).unwrap()
    }

    fn f32_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = f32::from_le_bytes(bytes[4*i..4*i+4].try_into().unwrap());
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f32_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let v = f32::from_le_bytes(bytes[4*i..4*i+4].try_into().unwrap());
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::bf16::from_bits(bits).to_f32();
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn bf16_bytes_to_f16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::bf16::from_bits(bits).to_f32();
            out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f16_bytes_to_bf16(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 2);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::f16::from_bits(bits).to_f32();
            out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        out
    }

    fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<u8> {
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2*i], bytes[2*i+1]]);
            let v = half::f16::from_bits(bits).to_f32();
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn f32_bytes_to_bf16_bytes(bytes: &[u8]) -> Vec<u8> {
        f32_bytes_to_bf16(bytes)
    }

    fn bf16_bytes_to_f16_bytes(bytes: &[u8]) -> Vec<u8> {
        bf16_bytes_to_f16(bytes)
    }

    fn f16_to_f32_val(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    use std::collections::BTreeMap;
}
