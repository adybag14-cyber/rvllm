// Enforces the per-crate and per-file LoC budgets from v3/IMPL_PLAN.md 1.1.
//   - each crate: <= 800 LoC (sum of src/**/*.rs, excluding blank + comment-only lines)
//   - each file:  <= 500 LoC
// If a crate/file needs to exceed, either split it or promote the limit explicitly
// with a comment `// loc-budget-override: <reason>` in the file header.

use std::fs;
use std::path::{Path, PathBuf};

const PER_CRATE: usize = 800;
const PER_FILE: usize = 500;

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn is_override(path: &Path) -> bool {
    fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .take(5)
                .any(|l| l.contains("loc-budget-override:"))
        })
        .unwrap_or(false)
}

fn count_effective(path: &Path) -> usize {
    let Ok(body) = fs::read_to_string(path) else {
        return 0;
    };
    body.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.is_empty() && !t.starts_with("//")
        })
        .count()
}

fn walk_rs(dir: &Path, acc: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk_rs(&p, acc);
        } else if p.extension().map(|e| e == "rs").unwrap_or(false) {
            acc.push(p);
        }
    }
}

#[test]
fn loc_budgets_per_file() {
    let dir = crates_dir();
    let mut violations = Vec::new();
    let mut files = Vec::new();
    for entry in fs::read_dir(&dir).expect("read crates/").flatten() {
        let src = entry.path().join("src");
        if src.exists() {
            walk_rs(&src, &mut files);
        }
    }
    for f in &files {
        if is_override(f) {
            continue;
        }
        let n = count_effective(f);
        if n > PER_FILE {
            violations.push(format!("{}: {n} lines (>{PER_FILE})", f.display()));
        }
    }
    assert!(
        violations.is_empty(),
        "per-file LoC violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn loc_budgets_per_crate() {
    let dir = crates_dir();
    let mut violations = Vec::new();
    for entry in fs::read_dir(&dir).expect("read crates/").flatten() {
        let c = entry.path();
        if !c.is_dir() {
            continue;
        }
        let src = c.join("src");
        if !src.exists() {
            continue;
        }
        let mut files = Vec::new();
        walk_rs(&src, &mut files);
        let total: usize = files
            .iter()
            .filter(|f| !is_override(f))
            .map(|f| count_effective(f))
            .sum();
        if total > PER_CRATE {
            let name = c
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_string();
            violations.push(format!("{name}: {total} lines (>{PER_CRATE})"));
        }
    }
    assert!(
        violations.is_empty(),
        "per-crate LoC violations:\n{}",
        violations.join("\n")
    );
}
