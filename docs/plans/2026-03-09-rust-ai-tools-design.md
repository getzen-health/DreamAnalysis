# Design: Rust AI Tools — nano-rag, llm-bench, prompt-rs

**Date**: 2026-03-09
**Author**: Sravya
**Goal**: Build 3 Rust-based AI developer tools inspired by Karpathy's formula —
minimal, working, educational — published to GitHub + Dev.to as a series.

---

## Why Rust + AI Developer Tools

Andrej Karpathy's most-starred repos share one formula:
> "Smallest complete thing that teaches the real thing — no magic, no abstractions."

The AI developer ecosystem is overwhelmingly Python. Rust fills an unexploited gap:
- 50–100× faster than Python equivalents
- Near-zero Rust AI tooling exists (first-mover advantage)
- Systems programmers entering AI actively look for Rust-native options
- Rust's type system enables guarantees Python cannot (compile-time prompt validation)

---

## Project 1: `nano-rag`

**Tagline**: The micrograd of RAG — 300 lines of Rust, no magic.

### What it does
End-to-end retrieval-augmented generation pipeline with every abstraction stripped:

```
text → chunks → embeddings (OpenAI or local) → cosine similarity store
     → top-K retrieval → prompt assembly → LLM call → response
```

### Architecture
- `src/chunk.rs` — text splitting (fixed-size + sentence-boundary)
- `src/embed.rs` — embedding via OpenAI API (or stub for local)
- `src/store.rs` — in-memory vector store, cosine similarity, top-K search
- `src/retriever.rs` — query → embed → retrieve → augment prompt
- `src/llm.rs` — HTTP call to OpenAI/Claude completion endpoint
- `src/main.rs` — CLI entrypoint: `nano-rag query "what is X?" --docs ./docs/`
- Total: ~300 lines across all files

### Why it gets stars
RAG is the #1 technique AI devs use. LangChain/LlamaIndex abstract every step.
This shows the whole pipeline in one readable Rust codebase — forkable, hackable,
no magic. Same reason micrograd got 15K stars: you understand all of it in one sitting.

### Dev.to post
*"I rewrote LangChain in 300 lines of Rust and here's what I found"*

---

## Project 2: `llm-bench`

**Tagline**: Benchmark OpenAI, Claude, Groq, Ollama on your actual prompts — in milliseconds.

### What it does
Rust CLI binary. Takes a prompt (or prompt file) and runs it across multiple
LLM providers in parallel, outputting a comparison table: latency, cost, tokens,
and a simple output quality signal.

```bash
llm-bench run --prompt "summarize this" --models gpt-4o,claude-3-5-sonnet,groq-llama3
# ┌─────────────────────┬──────────┬────────┬────────┬───────────────┐
# │ Model               │ Latency  │ Cost   │ Tokens │ Output        │
# ├─────────────────────┼──────────┼────────┼────────┼───────────────┤
# │ gpt-4o              │  1,240ms │ $0.003 │   412  │ "The doc..."  │
# │ claude-3-5-sonnet   │    890ms │ $0.002 │   398  │ "This doc..." │
# │ groq-llama3-70b     │    120ms │ $0.001 │   405  │ "The file..." │
# └─────────────────────┴──────────┴────────┴────────┴───────────────┘
```

### Architecture
- `src/providers/` — one module per provider (openai, anthropic, groq, ollama)
- `src/runner.rs` — parallel async runner (tokio), collects results
- `src/report.rs` — table formatting, cost calculation, token counting
- `src/cli.rs` — clap-based CLI (run, compare, cost-estimate subcommands)
- `src/config.rs` — API key management via env vars or `~/.llm-bench.toml`
- Binary: `llm-bench` — cross-compiled to Linux/macOS/Windows

### Why it gets stars
Every AI dev has this exact problem. No good tool exists. Rust makes it fast enough
to run in a pre-commit hook or CI pipeline. Like `hyperfine` (49K stars) but for LLMs.

### Dev.to post
*"I built the LLM benchmarking tool I always wanted (in Rust)"*

---

## Project 3: `prompt-rs`

**Tagline**: Type-safe LLM prompt templates. The Rust type system prevents prompt bugs at compile time.

### What it does
A Rust crate (`cargo add prompt-rs`) for building LLM prompts with compile-time
variable validation, injection prevention, and role-structured message builders.

```rust
use prompt_rs::prompt;

// compile error if {document} or {language} are not supplied
let p = prompt!("Summarize {document} in {language}");
let built = p.fill("document", &doc).fill("language", "English").build()?;

// Role-structured prompts (OpenAI chat format)
let messages = Chat::new()
    .system("You are a helpful assistant")
    .user(p)
    .build();
```

### Architecture
- `src/template.rs` — proc macro `prompt!` with compile-time variable parsing
- `src/builder.rs` — typed builder pattern, `fill()`, `build()` -> `Result<String>`
- `src/chat.rs` — role-structured message builder (system/user/assistant)
- `src/guard.rs` — injection detection (checks for common jailbreak patterns)
- Published to crates.io

### Why it gets stars
Rust developers entering AI want Rust-idiomatic primitives. No existing crate does
this with compile-time guarantees. Targets Candle, Burn, and rig ecosystem users.

### Dev.to post
*"Type-safe LLM prompts in Rust: catching prompt bugs before they happen"*

---

## Publishing Strategy

### GitHub
- 3 separate repos under `LakshmiSravyaVedantham/`
- Each repo: clean README with usage example, architecture diagram, install instructions
- Badge: `crates.io` version, CI status, license (MIT)
- Topics: `rust`, `llm`, `ai`, `rag`, `openai`, `claude`, `developer-tools`

### Dev.to Series
Publish as a 3-part series: *"Building AI Tools in Rust"*
- Part 1: nano-rag (deepest technical content, highest share potential)
- Part 2: llm-bench (most practically useful, broadest audience)
- Part 3: prompt-rs (most niche, Rust-specific audience)

Post gap: 1 week between each to sustain visibility.

### Cross-posting
- Hacker News Show HN for nano-rag (best fit — systems + AI overlap)
- r/rust + r/MachineLearning for all 3
- Twitter/X thread for each: Karpathy-style "I learned X by building Y"

---

## Success Metrics (3 months)

| Project | Stars target | crates.io downloads | Dev.to reads |
|---------|-------------|---------------------|--------------|
| nano-rag | 500+ | — | 5,000+ |
| llm-bench | 300+ | — | 3,000+ |
| prompt-rs | 150+ | 1,000+ | 2,000+ |

---

## Build Order

1. `llm-bench` first — fastest to build, most immediately useful, validates the Rust setup
2. `nano-rag` second — flagship educational project, most star potential
3. `prompt-rs` third — crate publication takes most polish

---

## Stack

- Rust stable (1.75+)
- `tokio` — async runtime
- `reqwest` — HTTP client
- `serde` / `serde_json` — JSON
- `clap` — CLI argument parsing
- `colored` / `comfy-table` — terminal output
- `proc-macro2` + `syn` + `quote` — for prompt-rs macros
- CI: GitHub Actions (cargo test + cargo clippy + cross-compile)
