//! Environment-variable whitelist per `v3/specs/02-config.md`.
//!
//! Every recognized `RVLLM_*` var is listed here. The runtime scans
//! `std::env::vars()` once at startup and any `RVLLM_*` not in this
//! list is a hard error (`ConfigError::UnknownEnvVar`). This catches
//! typos such as `RVLLM_NOGRAPH=1` that would otherwise be silently
//! ignored.

/// The whitelist. Alphabetical.
pub const ENV_WHITELIST: &[&str] = &[
    "RVLLM_AUTOTUNE_CACHE",
    "RVLLM_DEVICE",
    "RVLLM_KERNEL_DIR",
    "RVLLM_LOG",
    "RVLLM_NO_GRAPH",
];

/// Returns the first unknown `RVLLM_*` var found in the process env,
/// or `None` if every such var is in the whitelist.
pub fn first_unknown_rvllm_env() -> Option<String> {
    for (key, _) in std::env::vars() {
        if !key.starts_with("RVLLM_") {
            continue;
        }
        if !ENV_WHITELIST.contains(&key.as_str()) {
            return Some(key);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitelist_is_sorted_and_unique() {
        let mut sorted = ENV_WHITELIST.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), ENV_WHITELIST);
    }

    #[test]
    fn detects_unknown_env_in_process() {
        // Set one bad, one good; detect the bad one.
        std::env::set_var("RVLLM_DEFINITELY_NOT_REAL", "1");
        let bad = first_unknown_rvllm_env();
        std::env::remove_var("RVLLM_DEFINITELY_NOT_REAL");
        assert_eq!(bad.as_deref(), Some("RVLLM_DEFINITELY_NOT_REAL"));
    }
}
