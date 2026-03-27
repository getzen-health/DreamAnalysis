#!/usr/bin/env python3
"""
NeuralDreamWorkshop Research Agent
------------------------------------
Wakes every 20 minutes, searches arXiv for papers on one of the 5 key
accuracy-improvement areas, summarises findings via agent-hub (Groq),
and opens a GitHub issue with actionable recommendations.

Run once manually:   python3 research_agent.py
Run via cron:        */20 * * * * /usr/bin/python3 /path/to/research_agent.py
"""

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
ROUTER = Path.home() / ".claude/plugins/cache/claude-plugins-official/superpowers/5.0.5/skills/agent-hub/router.py"
GITHUB_REPO = "LakshmiSravyaVedantham/DreamAnalysis"
STATE_FILE = Path.home() / ".claude/agent-hub/research_agent_state.json"
ARXIV_API = "https://export.arxiv.org/api/query"
MAX_PAPERS = 5
GH_BIN = "/opt/homebrew/bin/gh"

# ── Research topics (cycle through in order) ───────────────────────────────────
TOPICS = [
    {
        "id": "temporal-eeg",
        "label": "Temporal EEG Modeling",
        "query": "EEG emotion recognition transformer temporal 4-channel consumer headband",
        "issue_label": "accuracy,ml,research",
        "context": (
            "Current system uses LightGBM on 41 hand-crafted features per 1-second epoch. "
            "We need temporal models (transformers, RNNs) that capture dynamics across epochs. "
            "Hardware: Muse 2, 4 channels (AF7, AF8, TP9, TP10), 256 Hz."
        ),
    },
    {
        "id": "domain-adaptation",
        "label": "Cross-Subject Domain Adaptation",
        "query": "EEG domain adaptation cross-subject emotion CORAL DANN transfer learning",
        "issue_label": "accuracy,ml,research",
        "context": (
            "Cross-subject CV is 71.52% on 9 datasets (DEAP, DREAMER, SEED-IV, etc.). "
            "Real Muse 2 deployment degrades further due to dry-electrode domain gap. "
            "Need domain adaptation to transfer from research EEG datasets to consumer headband."
        ),
    },
    {
        "id": "personalization",
        "label": "Few-Shot Personalization",
        "query": "EEG personalization few-shot meta-learning prototypical network online adaptation",
        "issue_label": "accuracy,ml,research",
        "context": (
            "Per-user baseline calibration (2-min resting EEG) already implemented. "
            "Online learner with SGD warm-start exists but is rudimentary. "
            "Need few-shot methods (5-10 labeled samples) to adapt the global model per user. "
            "Target: +15-20% accuracy after calibration."
        ),
    },
    {
        "id": "artifact-rejection",
        "label": "Artifact Rejection for Consumer EEG",
        "query": "EEG artifact removal consumer headband dry electrode muscle EMG ICA 4-channel",
        "issue_label": "signal-quality,research",
        "context": (
            "Current artifact rejection: ±75 µV threshold + kurtosis > 10. "
            "ICA is impossible on 4 channels. Gamma band is dominated by jaw EMG. "
            "Need better artifact rejection without requiring extra channels."
        ),
    },
    {
        "id": "multimodal-fusion",
        "label": "Multimodal Fusion (EEG + Voice + HRV)",
        "query": "multimodal emotion recognition EEG voice HRV fusion optimal transport alignment",
        "issue_label": "accuracy,ml,research",
        "context": (
            "System fuses EEG, voice (emotion2vec+), and health data (HRV, sleep, activity). "
            "Current fusion: late-stage weighted average. "
            "Need principled alignment (optimal transport, attention-based fusion) "
            "especially when modalities disagree."
        ),
    },
]


# ── State: track which topic to run next ───────────────────────────────────────
def load_state() -> dict:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"next_topic_index": 0, "issues_created": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── arXiv search ───────────────────────────────────────────────────────────────
def search_arxiv(query: str, max_results: int = MAX_PAPERS) -> list:
    """Return list of {title, authors, summary, url, published} dicts."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read()
    except Exception as e:
        print(f"[research-agent] arXiv fetch failed: {e}", file=sys.stderr)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        link_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        if title_el is None:
            continue
        papers.append({
            "title": title_el.text.strip().replace("\n", " "),
            "authors": ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""),
            "summary": (summary_el.text or "").strip().replace("\n", " ")[:400],
            "url": (link_el.text or "").strip(),
            "published": (published_el.text or "")[:10],
        })
    return papers


# ── agent-hub call ─────────────────────────────────────────────────────────────
def ask_agent_hub(prompt: str) -> str:
    """Route a prompt through agent-hub and return the text response."""
    result = subprocess.run(
        ["python3", str(ROUTER), "route", prompt, "--type", "general"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"agent-hub failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ── GitHub issue creation ──────────────────────────────────────────────────────
def create_github_issue(title: str, body: str, labels: str) -> str:
    """Create a GitHub issue and return its URL."""
    result = subprocess.run(
        [
            GH_BIN, "issue", "create",
            "--repo", GITHUB_REPO,
            "--title", title,
            "--body", body,
            "--label", labels,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh issue create failed: {result.stderr.strip()}")
    return result.stdout.strip()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"[research-agent] Starting at {datetime.now(timezone.utc).isoformat()}Z")

    state = load_state()
    idx = state["next_topic_index"] % len(TOPICS)
    topic = TOPICS[idx]

    print(f"[research-agent] Topic {idx + 1}/{len(TOPICS)}: {topic['label']}")

    # 1. Search arXiv
    papers = search_arxiv(topic["query"])
    if not papers:
        print("[research-agent] No papers found, skipping.", file=sys.stderr)
        return

    print(f"[research-agent] Found {len(papers)} papers")

    # 2. Format papers for the prompt
    papers_text = "\n\n".join(
        f"**{p['title']}** ({p['published']})\n{p['authors']}\n{p['summary']}\n{p['url']}"
        for p in papers
    )

    prompt = (
        f"You are a research assistant for NeuralDreamWorkshop, a real-time EEG emotion "
        f"recognition system using Muse 2 (4 channels, 256 Hz). Current accuracy: 71.52% "
        f"cross-subject CV on 9 datasets. "
        f"\n\nContext: {topic['context']}"
        f"\n\nHere are {len(papers)} recent arXiv papers on '{topic['label']}':\n\n{papers_text}"
        f"\n\nWrite a concise GitHub issue body (markdown) with:"
        f"\n1. **Summary** (2-3 sentences on the most relevant finding)"
        f"\n2. **Recommended Change** (specific code change or experiment to try)"
        f"\n3. **Expected Impact** (estimated accuracy improvement)"
        f"\n4. **Papers** (links to the most relevant 2-3)"
        f"\nBe specific and actionable. Max 400 words."
    )

    # 3. Ask agent-hub
    print("[research-agent] Asking agent-hub (Groq)...")
    try:
        issue_body = ask_agent_hub(prompt)
    except Exception as e:
        print(f"[research-agent] agent-hub error: {e}", file=sys.stderr)
        # Fallback: write raw paper list
        issue_body = f"**Auto-research: {topic['label']}**\n\n" + papers_text

    # Add metadata footer
    issue_body += (
        f"\n\n---\n*Auto-generated by research_agent.py at {datetime.now(timezone.utc).isoformat()}Z "
        f"via agent-hub (Groq llama-3.3-70b) · arXiv query: `{topic['query']}`*"
    )

    # 4. Create GitHub issue
    issue_title = f"[Research] {topic['label']}: {papers[0]['title'][:60]}..."
    print(f"[research-agent] Creating issue: {issue_title}")

    try:
        issue_url = create_github_issue(issue_title, issue_body, topic["issue_label"])
        print(f"[research-agent] Issue created: {issue_url}")
        state["issues_created"].append({"topic": topic["id"], "url": issue_url, "ts": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        print(f"[research-agent] GitHub issue creation failed: {e}", file=sys.stderr)

    # 5. Advance to next topic
    state["next_topic_index"] = (idx + 1) % len(TOPICS)
    save_state(state)
    print("[research-agent] Done.")


if __name__ == "__main__":
    main()
