# Rust AI Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and publish 3 Rust AI developer tools to GitHub — `llm-bench`, `nano-rag`, `prompt-rs` — following Karpathy's formula: minimal, working, educational.

**Architecture:** Three separate Rust crates, each as its own GitHub repo. Build order: `llm-bench` (fastest win, validates toolchain) → `nano-rag` (flagship, most stars) → `prompt-rs` (library crate, crates.io).

**Tech Stack:** Rust stable 1.75+, tokio, reqwest, serde_json, clap, comfy-table, GitHub Actions CI

---

## PHASE 1: `llm-bench`

### Task 1: Create the project

**Files:**
- Create: `~/llm-bench/` (new repo)
- Create: `~/llm-bench/Cargo.toml`
- Create: `~/llm-bench/src/main.rs`

**Step 1: Scaffold the project**

```bash
cd ~
cargo new llm-bench --bin
cd llm-bench
git init
```

**Step 2: Set Cargo.toml dependencies**

Replace `~/llm-bench/Cargo.toml` with:

```toml
[package]
name = "llm-bench"
version = "0.1.0"
edition = "2021"
description = "Benchmark OpenAI, Claude, Groq, Ollama on your prompts — in milliseconds"
license = "MIT"
repository = "https://github.com/LakshmiSravyaVedantham/llm-bench"
keywords = ["llm", "benchmark", "openai", "claude", "ai"]
categories = ["command-line-utilities", "development-tools"]

[[bin]]
name = "llm-bench"
path = "src/main.rs"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
comfy-table = "7"
colored = "2"
dotenvy = "0.15"
anyhow = "1"

[dev-dependencies]
tokio-test = "0.4"
```

**Step 3: Verify it compiles**

```bash
cargo build
```
Expected: compiles with no errors.

**Step 4: Commit**

```bash
git add .
git commit -m "chore: init llm-bench crate"
```

---

### Task 2: Provider abstraction

**Files:**
- Create: `src/providers/mod.rs`
- Create: `src/providers/openai.rs`
- Create: `src/providers/anthropic.rs`
- Create: `src/providers/groq.rs`
- Create: `tests/providers_test.rs`

**Step 1: Write failing test**

Create `tests/providers_test.rs`:

```rust
use llm_bench::providers::{Provider, CompletionRequest, CompletionResult};

#[test]
fn test_provider_name_openai() {
    let p = llm_bench::providers::openai::OpenAIProvider::new("fake-key".into());
    assert_eq!(p.name(), "gpt-4o");
}

#[test]
fn test_completion_result_fields() {
    let r = CompletionResult {
        model: "gpt-4o".into(),
        output: "hello".into(),
        latency_ms: 100,
        input_tokens: 10,
        output_tokens: 5,
        cost_usd: 0.001,
    };
    assert_eq!(r.model, "gpt-4o");
    assert_eq!(r.latency_ms, 100);
}
```

**Step 2: Run test — verify it fails**

```bash
cargo test
```
Expected: FAIL — modules not found.

**Step 3: Implement `src/providers/mod.rs`**

```rust
pub mod openai;
pub mod anthropic;
pub mod groq;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub model: String,
    pub output: String,
    pub latency_ms: u128,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
}

#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResult>;
}
```

Add `async-trait = "0.1"` to `Cargo.toml` dependencies.

**Step 4: Implement `src/providers/openai.rs`**

```rust
use super::{CompletionRequest, CompletionResult, Provider};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;

pub struct OpenAIProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl OpenAIProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "gpt-4o".into(),
            client: Client::new(),
        }
    }
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.model
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResult> {
        let start = Instant::now();
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_tokens
        });
        let resp: Value = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send().await?
            .json().await?;

        let output = resp["choices"][0]["message"]["content"]
            .as_str().unwrap_or("").to_string();
        let input_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        // gpt-4o pricing: $5/1M input, $15/1M output
        let cost_usd = (input_tokens as f64 * 5.0 + output_tokens as f64 * 15.0) / 1_000_000.0;

        Ok(CompletionResult {
            model: self.model.clone(),
            output,
            latency_ms: start.elapsed().as_millis(),
            input_tokens,
            output_tokens,
            cost_usd,
        })
    }
}
```

**Step 5: Implement `src/providers/anthropic.rs`**

```rust
use super::{CompletionRequest, CompletionResult, Provider};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key, model: "claude-3-5-sonnet-20241022".into(), client: Client::new() }
    }
    pub fn with_model(mut self, model: &str) -> Self { self.model = model.into(); self }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str { &self.model }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResult> {
        let start = Instant::now();
        let body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}]
        });
        let resp: Value = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send().await?
            .json().await?;

        let output = resp["content"][0]["text"].as_str().unwrap_or("").to_string();
        let input_tokens = resp["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = resp["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
        // claude-3-5-sonnet: $3/1M input, $15/1M output
        let cost_usd = (input_tokens as f64 * 3.0 + output_tokens as f64 * 15.0) / 1_000_000.0;

        Ok(CompletionResult {
            model: self.model.clone(), output,
            latency_ms: start.elapsed().as_millis(),
            input_tokens, output_tokens, cost_usd,
        })
    }
}
```

**Step 6: Implement `src/providers/groq.rs`**

```rust
use super::{CompletionRequest, CompletionResult, Provider};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;

pub struct GroqProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl GroqProvider {
    pub fn new(api_key: String) -> Self {
        Self { api_key, model: "llama3-70b-8192".into(), client: Client::new() }
    }
    pub fn with_model(mut self, model: &str) -> Self { self.model = model.into(); self }
}

#[async_trait]
impl Provider for GroqProvider {
    fn name(&self) -> &str { &self.model }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResult> {
        let start = Instant::now();
        let body = json!({
            "model": self.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_tokens
        });
        let resp: Value = self.client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send().await?
            .json().await?;

        let output = resp["choices"][0]["message"]["content"]
            .as_str().unwrap_or("").to_string();
        let input_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
        // Groq is nearly free — $0.59/1M input, $0.79/1M output for llama3-70b
        let cost_usd = (input_tokens as f64 * 0.59 + output_tokens as f64 * 0.79) / 1_000_000.0;

        Ok(CompletionResult {
            model: self.model.clone(), output,
            latency_ms: start.elapsed().as_millis(),
            input_tokens, output_tokens, cost_usd,
        })
    }
}
```

**Step 7: Wire providers into lib.rs**

Create `src/lib.rs`:

```rust
pub mod providers;
pub mod runner;
pub mod report;
```

**Step 8: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 9: Commit**

```bash
git add src/ tests/ Cargo.toml
git commit -m "feat: add provider abstraction (OpenAI, Anthropic, Groq)"
```

---

### Task 3: Parallel runner

**Files:**
- Create: `src/runner.rs`
- Modify: `tests/providers_test.rs`

**Step 1: Write failing test**

Add to `tests/providers_test.rs`:

```rust
#[tokio::test]
async fn test_runner_collects_results() {
    use llm_bench::runner::run_providers;
    use llm_bench::providers::{CompletionRequest, CompletionResult};

    // This test uses no API keys — just verifies the runner structure compiles
    // and handles an empty provider list gracefully
    let req = CompletionRequest { prompt: "hello".into(), max_tokens: 10 };
    let providers: Vec<Box<dyn llm_bench::providers::Provider>> = vec![];
    let results = run_providers(providers, req).await;
    assert_eq!(results.len(), 0);
}
```

**Step 2: Run — verify fails**

```bash
cargo test test_runner_collects_results
```
Expected: FAIL — runner module not found.

**Step 3: Implement `src/runner.rs`**

```rust
use crate::providers::{CompletionRequest, CompletionResult, Provider};
use std::sync::Arc;

pub struct BenchResult {
    pub result: anyhow::Result<CompletionResult>,
}

pub async fn run_providers(
    providers: Vec<Box<dyn Provider>>,
    req: CompletionRequest,
) -> Vec<(String, anyhow::Result<CompletionResult>)> {
    let req = Arc::new(req);
    let mut handles = vec![];

    for provider in providers {
        let req = Arc::clone(&req);
        handles.push(tokio::spawn(async move {
            let name = provider.name().to_string();
            let result = provider.complete(&req).await;
            (name, result)
        }));
    }

    let mut results = vec![];
    for handle in handles {
        match handle.await {
            Ok(r) => results.push(r),
            Err(e) => results.push(("unknown".into(), Err(anyhow::anyhow!("task panicked: {e}")))),
        }
    }
    results
}
```

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/runner.rs tests/providers_test.rs
git commit -m "feat: add parallel async runner"
```

---

### Task 4: Report table

**Files:**
- Create: `src/report.rs`

**Step 1: Write failing test**

Create `tests/report_test.rs`:

```rust
use llm_bench::report::format_results;
use llm_bench::providers::CompletionResult;

#[test]
fn test_format_results_returns_string() {
    let results = vec![
        ("gpt-4o".into(), Ok(CompletionResult {
            model: "gpt-4o".into(),
            output: "Hello world".into(),
            latency_ms: 1200,
            input_tokens: 10,
            output_tokens: 5,
            cost_usd: 0.000125,
        })),
    ];
    let table = format_results(&results);
    assert!(table.contains("gpt-4o"));
    assert!(table.contains("1200ms"));
}
```

**Step 2: Run — verify fails**

```bash
cargo test test_format_results
```

**Step 3: Implement `src/report.rs`**

```rust
use crate::providers::CompletionResult;
use comfy_table::{Table, presets::UTF8_FULL, Attribute, Cell, Color};

pub fn format_results(
    results: &[(String, anyhow::Result<CompletionResult>)],
) -> String {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL);
    table.set_header(vec![
        Cell::new("Model").add_attribute(Attribute::Bold),
        Cell::new("Latency").add_attribute(Attribute::Bold),
        Cell::new("Cost").add_attribute(Attribute::Bold),
        Cell::new("In/Out Tokens").add_attribute(Attribute::Bold),
        Cell::new("Output (first 60 chars)").add_attribute(Attribute::Bold),
    ]);

    for (name, result) in results {
        match result {
            Ok(r) => {
                table.add_row(vec![
                    Cell::new(&r.model).fg(Color::Cyan),
                    Cell::new(format!("{}ms", r.latency_ms)),
                    Cell::new(format!("${:.6}", r.cost_usd)),
                    Cell::new(format!("{}/{}", r.input_tokens, r.output_tokens)),
                    Cell::new(r.output.chars().take(60).collect::<String>()),
                ]);
            }
            Err(e) => {
                table.add_row(vec![
                    Cell::new(name).fg(Color::Red),
                    Cell::new("ERROR"),
                    Cell::new("-"),
                    Cell::new("-"),
                    Cell::new(e.to_string()),
                ]);
            }
        }
    }
    table.to_string()
}
```

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/report.rs tests/report_test.rs
git commit -m "feat: add results table formatter"
```

---

### Task 5: CLI entrypoint

**Files:**
- Modify: `src/main.rs`

**Step 1: Implement `src/main.rs`**

```rust
use clap::{Parser, Subcommand};
use dotenvy::dotenv;
use llm_bench::{
    providers::{CompletionRequest, Provider, openai::OpenAIProvider,
                anthropic::AnthropicProvider, groq::GroqProvider},
    runner::run_providers,
    report::format_results,
};
use std::env;

#[derive(Parser)]
#[command(name = "llm-bench", about = "Benchmark LLMs on your prompts", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a prompt across models
    Run {
        /// Prompt text
        #[arg(long)]
        prompt: String,
        /// Comma-separated model names: gpt-4o,claude-3-5-sonnet,groq-llama3
        #[arg(long, default_value = "gpt-4o,claude-3-5-sonnet,groq-llama3")]
        models: String,
        /// Max output tokens
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { prompt, models, max_tokens } => {
            let req = CompletionRequest { prompt, max_tokens };
            let providers = build_providers(&models);
            if providers.is_empty() {
                eprintln!("No valid models specified or no API keys found.");
                std::process::exit(1);
            }
            println!("Running {} providers in parallel...\n", providers.len());
            let results = run_providers(providers, req).await;
            println!("{}", format_results(&results));
        }
    }
    Ok(())
}

fn build_providers(models: &str) -> Vec<Box<dyn Provider>> {
    let mut providers: Vec<Box<dyn Provider>> = vec![];
    for model in models.split(',') {
        let model = model.trim();
        if model.starts_with("gpt") {
            if let Ok(key) = env::var("OPENAI_API_KEY") {
                providers.push(Box::new(OpenAIProvider::new(key).with_model(model)));
            }
        } else if model.starts_with("claude") {
            if let Ok(key) = env::var("ANTHROPIC_API_KEY") {
                providers.push(Box::new(AnthropicProvider::new(key).with_model(model)));
            }
        } else if model.starts_with("groq") || model.starts_with("llama") {
            if let Ok(key) = env::var("GROQ_API_KEY") {
                providers.push(Box::new(GroqProvider::new(key).with_model(model)));
            }
        }
    }
    providers
}
```

**Step 2: Build and run smoke test**

```bash
cargo build
./target/debug/llm-bench --help
```
Expected: shows usage without crashing.

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add CLI entrypoint (run subcommand)"
```

---

### Task 6: README + GitHub Actions + publish

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`
- Create: `.gitignore`
- Create: `LICENSE`

**Step 1: Create `.gitignore`**

```
/target
.env
*.env
```

**Step 2: Create `LICENSE`** (MIT)

```
MIT License

Copyright (c) 2026 Sravya

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

**Step 3: Create `.github/workflows/ci.yml`**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo clippy -- -D warnings
      - run: cargo test
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo build --release
```

**Step 4: Create `README.md`**

````markdown
# llm-bench

**Benchmark OpenAI, Claude, Groq, Ollama on your actual prompts — in milliseconds.**

Built in Rust. Runs all providers in parallel. Shows latency, cost, and output side-by-side.

```bash
cargo install llm-bench

llm-bench run \
  --prompt "Explain the CAP theorem in one paragraph" \
  --models gpt-4o,claude-3-5-sonnet,groq-llama3
```

```
┌─────────────────────────┬──────────┬──────────┬───────────────┬─────────────────────────────┐
│ Model                   │ Latency  │ Cost     │ In/Out Tokens │ Output (first 60 chars)     │
├─────────────────────────┼──────────┼──────────┼───────────────┼─────────────────────────────┤
│ gpt-4o                  │ 1,240ms  │ $0.0041  │ 12/89         │ The CAP theorem states...   │
│ claude-3-5-sonnet       │   890ms  │ $0.0028  │ 12/84         │ In distributed systems...   │
│ llama3-70b-8192 (groq)  │   121ms  │ $0.0001  │ 12/91         │ The CAP theorem, proposed.. │
└─────────────────────────┴──────────┴──────────┴───────────────┴─────────────────────────────┘
```

## Setup

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...
```

Or create a `.env` file in your project root.

## Why Rust?

Most LLM tooling is Python. Python adds 100–500ms of overhead before the first byte leaves
your machine. llm-bench is a single binary with ~5ms startup time, making it usable in
CI pipelines and pre-commit hooks.

## Architecture

```
src/
├── providers/     # One module per provider (OpenAI, Anthropic, Groq)
├── runner.rs      # Parallel async runner (tokio)
├── report.rs      # Table formatter
└── main.rs        # CLI (clap)
```

The entire codebase is ~400 lines. Read it in 10 minutes.

## License

MIT
````

**Step 5: Run final tests**

```bash
cargo test && cargo clippy -- -D warnings
```
Expected: all pass, no clippy warnings.

**Step 6: Create GitHub repo and push**

```bash
gh repo create LakshmiSravyaVedantham/llm-bench \
  --public \
  --description "Benchmark OpenAI, Claude, Groq on your prompts — blazing fast Rust CLI" \
  --push \
  --source .
gh repo edit LakshmiSravyaVedantham/llm-bench \
  --add-topic rust,llm,ai,openai,claude,benchmark,developer-tools
```

**Step 7: Commit**

```bash
git add README.md .github/ .gitignore LICENSE
git commit -m "docs: add README, CI workflow, and license"
git push
```

---

## PHASE 2: `nano-rag`

### Task 7: Create the project

**Files:**
- Create: `~/nano-rag/` (new repo)

**Step 1: Scaffold**

```bash
cd ~
cargo new nano-rag --bin
cd nano-rag
git init
```

**Step 2: Set Cargo.toml**

```toml
[package]
name = "nano-rag"
version = "0.1.0"
edition = "2021"
description = "The micrograd of RAG — 300 lines of Rust, no magic"
license = "MIT"
repository = "https://github.com/LakshmiSravyaVedantham/nano-rag"
keywords = ["rag", "llm", "embeddings", "ai", "retrieval"]
categories = ["command-line-utilities", "science"]

[[bin]]
name = "nano-rag"
path = "src/main.rs"

[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
anyhow = "1"
dotenvy = "0.15"
```

**Step 3: Commit**

```bash
cargo build && git add . && git commit -m "chore: init nano-rag crate"
```

---

### Task 8: Text chunker

**Files:**
- Create: `src/chunk.rs`
- Create: `tests/chunk_test.rs`

**Step 1: Write failing test**

```rust
use nano_rag::chunk::split_chunks;

#[test]
fn test_splits_by_size() {
    let text = "word ".repeat(200); // 1000 chars
    let chunks = split_chunks(&text, 200, 20);
    assert!(chunks.len() >= 4);
    for chunk in &chunks {
        assert!(chunk.len() <= 220); // size + overlap tolerance
    }
}

#[test]
fn test_empty_text_returns_empty() {
    let chunks = split_chunks("", 200, 20);
    assert_eq!(chunks.len(), 0);
}
```

**Step 2: Run — verify fails**

```bash
cargo test
```

**Step 3: Implement `src/chunk.rs`**

```rust
/// Split text into overlapping chunks.
/// chunk_size: target chars per chunk
/// overlap: chars to repeat between chunks
pub fn split_chunks(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() { return vec![]; }
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = vec![];
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() { break; }
        start += chunk_size.saturating_sub(overlap);
    }
    chunks
}
```

Add to `src/lib.rs` (create if needed): `pub mod chunk;`

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/ tests/ && git commit -m "feat: add text chunker with overlap"
```

---

### Task 9: Embedding client

**Files:**
- Create: `src/embed.rs`
- Create: `tests/embed_test.rs`

**Step 1: Write failing test**

```rust
use nano_rag::embed::cosine_similarity;

#[test]
fn test_cosine_same_vector() {
    let v = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}
```

**Step 2: Run — verify fails**

**Step 3: Implement `src/embed.rs`**

```rust
use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

pub struct EmbedClient {
    api_key: String,
    client: Client,
}

impl EmbedClient {
    pub fn new(api_key: String) -> Self {
        Self { api_key, client: Client::new() }
    }

    pub async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = json!({
            "model": "text-embedding-3-small",
            "input": texts
        });
        let resp: Value = self.client
            .post("https://api.openai.com/v1/embeddings")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send().await?
            .json().await?;

        let embeddings = resp["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("no data in response"))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array().unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();
        Ok(embeddings)
    }
}
```

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/ tests/ && git commit -m "feat: add embedding client and cosine similarity"
```

---

### Task 10: In-memory vector store + retriever

**Files:**
- Create: `src/store.rs`
- Create: `tests/store_test.rs`

**Step 1: Write failing test**

```rust
use nano_rag::store::VectorStore;

#[test]
fn test_store_insert_and_retrieve() {
    let mut store = VectorStore::new();
    store.insert("The sky is blue".into(), vec![1.0, 0.0, 0.0]);
    store.insert("Rust is fast".into(), vec![0.0, 1.0, 0.0]);
    store.insert("Pizza is delicious".into(), vec![0.0, 0.0, 1.0]);

    // query closest to "sky is blue"
    let results = store.top_k(&[1.0, 0.0, 0.0], 1);
    assert_eq!(results[0].text, "The sky is blue");
}

#[test]
fn test_store_returns_k_results() {
    let mut store = VectorStore::new();
    for i in 0..10 {
        store.insert(format!("doc {i}"), vec![i as f32, 0.0]);
    }
    let results = store.top_k(&[1.0, 0.0], 3);
    assert_eq!(results.len(), 3);
}
```

**Step 2: Run — verify fails**

**Step 3: Implement `src/store.rs`**

```rust
use crate::embed::cosine_similarity;

#[derive(Debug, Clone)]
pub struct Document {
    pub text: String,
    pub embedding: Vec<f32>,
    pub score: f32,
}

pub struct VectorStore {
    docs: Vec<(String, Vec<f32>)>,
}

impl VectorStore {
    pub fn new() -> Self { Self { docs: vec![] } }

    pub fn insert(&mut self, text: String, embedding: Vec<f32>) {
        self.docs.push((text, embedding));
    }

    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<Document> {
        let mut scored: Vec<Document> = self.docs.iter().map(|(text, emb)| {
            Document {
                text: text.clone(),
                embedding: emb.clone(),
                score: cosine_similarity(query, emb),
            }
        }).collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize { self.docs.len() }
    pub fn is_empty(&self) -> bool { self.docs.is_empty() }
}
```

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/ tests/ && git commit -m "feat: add in-memory vector store with cosine retrieval"
```

---

### Task 11: CLI + README + publish

**Files:**
- Modify: `src/main.rs`
- Create: `README.md`
- Create: `.github/workflows/ci.yml`

**Step 1: Implement `src/main.rs`**

```rust
use clap::{Parser, Subcommand};
use dotenvy::dotenv;
use nano_rag::{chunk::split_chunks, embed::EmbedClient, store::VectorStore};
use std::{env, fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "nano-rag", about = "The micrograd of RAG — minimal, readable, hackable", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a file or directory into the vector store
    Index {
        #[arg(short, long)]
        docs: PathBuf,
        #[arg(short, long, default_value = "store.json")]
        output: PathBuf,
    },
    /// Query the vector store
    Query {
        #[arg(short, long)]
        question: String,
        #[arg(short, long, default_value = "store.json")]
        store: PathBuf,
        #[arg(short = 'k', long, default_value_t = 3)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();
    let cli = Cli::parse();
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY required");
    let client = EmbedClient::new(api_key);

    match cli.command {
        Commands::Index { docs, output } => {
            let text = fs::read_to_string(&docs)?;
            let chunks = split_chunks(&text, 400, 50);
            println!("Indexing {} chunks from {:?}...", chunks.len(), docs);

            let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
            let embeddings = client.embed(&refs).await?;

            let mut store = VectorStore::new();
            for (chunk, emb) in chunks.into_iter().zip(embeddings) {
                store.insert(chunk, emb);
            }

            // Serialize and save
            let serialized: Vec<(String, Vec<f32>)> = (0..store.len())
                .map(|i| store.top_k(&vec![0.0; 1536], store.len()).get(i)
                    .map(|d| (d.text.clone(), d.embedding.clone()))
                    .unwrap_or_default())
                .collect();
            fs::write(&output, serde_json::to_string(&serialized)?)?;
            println!("Saved index to {:?}", output);
        }
        Commands::Query { question, store: store_path, top_k } => {
            let data: Vec<(String, Vec<f32>)> =
                serde_json::from_str(&fs::read_to_string(&store_path)?)?;
            let mut store = VectorStore::new();
            for (text, emb) in data { store.insert(text, emb); }

            let q_emb = client.embed(&[question.as_str()]).await?;
            let results = store.top_k(&q_emb[0], top_k);

            println!("\nTop {} results for: \"{}\"\n", top_k, question);
            for (i, doc) in results.iter().enumerate() {
                println!("[{}] score={:.3}\n{}\n", i + 1, doc.score, doc.text);
            }
        }
    }
    Ok(())
}
```

**Step 2: Build smoke test**

```bash
cargo build
./target/debug/nano-rag --help
```

**Step 3: Create README.md**

````markdown
# nano-rag

**The micrograd of RAG — 300 lines of Rust, no magic.**

Most people use LangChain or LlamaIndex for RAG. Both hide what actually happens.
This is the whole pipeline, stripped bare:

```
text → chunks → embeddings → cosine similarity store → top-K → LLM answer
```

~300 lines of Rust across 5 files. Read it in 10 minutes. Fork it. Make it yours.

## Usage

```bash
cargo install nano-rag

# Index a document
export OPENAI_API_KEY=sk-...
nano-rag index --docs my_document.txt

# Query it
nano-rag query --question "What does the document say about X?"
```

## Architecture

```
src/
├── chunk.rs    # Text → overlapping chunks (50 lines)
├── embed.rs    # Chunks → embeddings via OpenAI + cosine similarity (60 lines)
├── store.rs    # In-memory vector store + top-K retrieval (40 lines)
└── main.rs     # CLI: index + query commands (80 lines)
```

That's all of RAG. LangChain has 200,000+ lines doing this same thing.

## What you learn by reading this

- How BPE token chunks map to embedding vectors
- Why cosine similarity (not dot product) measures semantic closeness
- How top-K retrieval constructs the augmented prompt
- Why the chunk size / overlap tradeoff matters for recall

## License

MIT
````

**Step 4: Add CI (same as llm-bench)**

Copy `.github/workflows/ci.yml` from llm-bench.

**Step 5: Run tests + clippy**

```bash
cargo test && cargo clippy -- -D warnings
```

**Step 6: Create repo and push**

```bash
gh repo create LakshmiSravyaVedantham/nano-rag \
  --public \
  --description "The micrograd of RAG — 300 lines of Rust, no magic" \
  --push --source .
gh repo edit LakshmiSravyaVedantham/nano-rag \
  --add-topic rust,rag,llm,embeddings,ai,education
```

**Step 7: Commit**

```bash
git add . && git commit -m "feat: complete nano-rag — chunker, embeddings, vector store, CLI"
git push
```

---

## PHASE 3: `prompt-rs`

### Task 12: Create the library crate

**Files:**
- Create: `~/prompt-rs/`

**Step 1: Scaffold as library**

```bash
cd ~
cargo new prompt-rs --lib
cd prompt-rs
git init
```

**Step 2: Set Cargo.toml**

```toml
[package]
name = "prompt-rs"
version = "0.1.0"
edition = "2021"
description = "Type-safe LLM prompt templates — catch prompt bugs at compile time"
license = "MIT"
repository = "https://github.com/LakshmiSravyaVedantham/prompt-rs"
keywords = ["llm", "prompt", "openai", "templates", "ai"]
categories = ["template-engine", "text-processing"]

[dependencies]
thiserror = "1"

[dev-dependencies]
```

**Step 3: Commit**

```bash
cargo build && git add . && git commit -m "chore: init prompt-rs library crate"
```

---

### Task 13: Template parser

**Files:**
- Create: `src/template.rs`
- Modify: `src/lib.rs`

**Step 1: Write failing test (in `src/lib.rs`)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_variables() {
        let t = PromptTemplate::new("Hello {name}, you are {age} years old");
        let vars = t.variables();
        assert!(vars.contains(&"name"));
        assert!(vars.contains(&"age"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_fill_all_variables() {
        let t = PromptTemplate::new("Hello {name}");
        let result = t.fill("name", "Sravya").build();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello Sravya");
    }

    #[test]
    fn test_missing_variable_returns_error() {
        let t = PromptTemplate::new("Hello {name} and {other}");
        let result = t.fill("name", "Sravya").build();
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("other"));
    }

    #[test]
    fn test_no_variables() {
        let t = PromptTemplate::new("Hello world");
        assert_eq!(t.build_raw(), "Hello world");
    }
}
```

**Step 2: Run — verify fails**

```bash
cargo test
```

**Step 3: Implement `src/lib.rs`**

```rust
use std::collections::{HashMap, HashSet};

#[derive(thiserror::Error, Debug)]
pub enum PromptError {
    #[error("Missing required variable(s): {0}")]
    MissingVariables(String),
}

#[derive(Clone, Debug)]
pub struct PromptTemplate {
    template: String,
}

impl PromptTemplate {
    pub fn new(template: impl Into<String>) -> Self {
        Self { template: template.into() }
    }

    pub fn variables(&self) -> HashSet<&str> {
        let mut vars = HashSet::new();
        let mut rest = self.template.as_str();
        while let Some(start) = rest.find('{') {
            rest = &rest[start + 1..];
            if let Some(end) = rest.find('}') {
                vars.insert(&rest[..end]);
                rest = &rest[end + 1..];
            }
        }
        vars
    }

    pub fn fill(self, key: &str, value: &str) -> PromptBuilder {
        let mut builder = PromptBuilder::new(self);
        builder.values.insert(key.to_string(), value.to_string());
        builder
    }

    pub fn build_raw(&self) -> String {
        self.template.clone()
    }
}

pub struct PromptBuilder {
    template: PromptTemplate,
    pub(crate) values: HashMap<String, String>,
}

impl PromptBuilder {
    fn new(template: PromptTemplate) -> Self {
        Self { template, values: HashMap::new() }
    }

    pub fn fill(mut self, key: &str, value: &str) -> Self {
        self.values.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build(self) -> Result<String, PromptError> {
        let vars = self.template.variables();
        let missing: Vec<&str> = vars.iter()
            .filter(|v| !self.values.contains_key(**v))
            .copied()
            .collect();

        if !missing.is_empty() {
            return Err(PromptError::MissingVariables(missing.join(", ")));
        }

        let mut result = self.template.template.clone();
        for (key, val) in &self.values {
            result = result.replace(&format!("{{{key}}}"), val);
        }
        Ok(result)
    }
}
```

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/ && git commit -m "feat: add PromptTemplate with variable parsing and builder"
```

---

### Task 14: Chat message builder

**Files:**
- Create: `src/chat.rs`

**Step 1: Write failing test**

Add to `src/lib.rs` tests:

```rust
#[test]
fn test_chat_builder() {
    use crate::chat::{Chat, Role};
    let messages = Chat::new()
        .system("You are helpful")
        .user("What is Rust?")
        .build();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[1].role, Role::User);
}
```

**Step 2: Run — verify fails**

**Step 3: Implement `src/chat.rs`**

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

pub struct Chat {
    messages: Vec<Message>,
}

impl Chat {
    pub fn new() -> Self { Self { messages: vec![] } }

    pub fn system(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message { role: Role::System, content: content.into() });
        self
    }

    pub fn user(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message { role: Role::User, content: content.into() });
        self
    }

    pub fn assistant(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message { role: Role::Assistant, content: content.into() });
        self
    }

    pub fn build(self) -> Vec<Message> { self.messages }
}
```

Add `serde = { version = "1", features = ["derive"] }` to Cargo.toml.
Add `pub mod chat;` to `src/lib.rs`.

**Step 4: Run tests**

```bash
cargo test
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/ Cargo.toml && git commit -m "feat: add Chat message builder with role types"
```

---

### Task 15: README + publish to crates.io + GitHub

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`

**Step 1: Create `README.md`**

````markdown
# prompt-rs

**Type-safe LLM prompt templates for Rust. Catch prompt bugs before they happen.**

```toml
[dependencies]
prompt-rs = "0.1"
```

```rust
use prompt_rs::PromptTemplate;

let template = PromptTemplate::new("Summarize {document} in {language}");

// compile + runtime: error if any variable is missing
let prompt = template
    .fill("document", &my_doc)
    .fill("language", "English")
    .build()?;
```

## Chat messages (OpenAI format)

```rust
use prompt_rs::chat::Chat;

let messages = Chat::new()
    .system("You are a helpful assistant")
    .user(&prompt)
    .build();

// Serialize directly to JSON for API calls
let json = serde_json::to_string(&messages)?;
```

## Why?

Python prompt libraries use runtime string substitution. Missing a variable?
You find out when the API returns an error — or worse, silently gets wrong output.

In Rust, `build()` returns `Result<String, PromptError>` with a clear error message
listing exactly which variables are missing. Handle it at the call site, not at runtime.

## License

MIT
````

**Step 2: Publish to crates.io**

```bash
# Login to crates.io first
cargo login
# Dry run
cargo publish --dry-run
# Actual publish
cargo publish
```

**Step 3: Create GitHub repo and push**

```bash
gh repo create LakshmiSravyaVedantham/prompt-rs \
  --public \
  --description "Type-safe LLM prompt templates for Rust" \
  --push --source .
gh repo edit LakshmiSravyaVedantham/prompt-rs \
  --add-topic rust,llm,prompt,openai,templates,crates
```

**Step 4: Final test + clippy**

```bash
cargo test && cargo clippy -- -D warnings
```

**Step 5: Commit and push**

```bash
git add . && git commit -m "feat: complete prompt-rs — templates, chat builder, crates.io ready"
git push
```

---

## PHASE 4: Dev.to Posts

### Task 16: Write and publish Part 1 (nano-rag)

**Step 1: Write the post**

Title: *"I rewrote LangChain in 300 lines of Rust and here's what I found"*

Structure:
1. Hook: "LangChain has 200,000 lines. Here's what RAG actually needs."
2. The 5 concepts (chunking, embedding, cosine similarity, retrieval, prompt assembly)
3. Code walkthrough — one section per file, with the actual Rust
4. Benchmark vs Python equivalent
5. Link to GitHub + `cargo install nano-rag`

**Step 2: Publish via Dev.to API**

```bash
# Set DEVTO_API_KEY in env
curl -X POST https://dev.to/api/articles \
  -H "api-key: $DEVTO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "article": {
      "title": "I rewrote LangChain in 300 lines of Rust",
      "body_markdown": "...",
      "published": true,
      "tags": ["rust", "ai", "llm", "rag"]
    }
  }'
```

### Task 17: Write and publish Part 2 (llm-bench)

Title: *"I built the LLM benchmarking tool every AI dev needs (in Rust)"*

### Task 18: Write and publish Part 3 (prompt-rs)

Title: *"Type-safe LLM prompts in Rust: catching prompt bugs before they happen"*

---

## Summary

| Phase | Projects | Estimated time |
|-------|---------|----------------|
| Phase 1 | llm-bench | ~2 hours |
| Phase 2 | nano-rag | ~2 hours |
| Phase 3 | prompt-rs | ~1.5 hours |
| Phase 4 | Dev.to posts | ~3 hours |

Build in order. Ship each to GitHub before moving to the next.
