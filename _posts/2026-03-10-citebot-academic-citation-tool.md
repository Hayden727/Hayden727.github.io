---
layout: post
title: "CiteBot: Automating Academic Citations with LLM + NLP Fusion"
date: 2026-03-10
categories: [tools]
excerpt: "Building an intelligent LaTeX citation assistant that fuses LLM semantic understanding with NLP term extraction to find and format academic references."
---

# CiteBot: Automating Academic Citations with LLM + NLP Fusion

Every researcher knows the pain: you've written a paper, and now you need to find 30-100 relevant references, format them as BibTeX, and insert `\cite{}` commands in the right places. It's tedious, error-prone, and takes hours.

[CiteBot](https://github.com/Hayden727/CiteBot) automates this entire workflow. Give it a `.tex` file, and it produces a complete `references.bib` — with optional `\cite{}` insertion. The interesting engineering challenge is making the reference *selection* actually good.

## The Pipeline

```
.tex file(s)
  → Parse LaTeX (title, abstract, sections)
  → Extract keywords (LLM + NLP fusion)
  → Search academic databases (5 sources, async)
  → Deduplicate + rank (composite scoring)
  → Generate BibTeX (DOI negotiation)
  → Insert \cite{} (fuzzy matching, optional)
```

Each stage is its own module, connected by immutable dataclasses.

## The Keyword Extraction Problem

The core challenge: given a LaTeX document, extract keywords that will *actually find relevant papers* in academic databases.

This sounds easy. It's not.

### Why LLMs Alone Aren't Enough

An LLM reads your paper and produces semantically meaningful concepts: "domain-specific compilation", "memory hierarchy optimization", "cost-model-driven scheduling". These are great for understanding the paper — but they make terrible search queries. Academic databases match on terms that appear in paper titles and abstracts, not on high-level concepts.

### Why NLP Alone Isn't Enough

Statistical keyword extractors (KeyBERT, YAKE, spaCy) find high-frequency terms: "MLIR", "LLVM", "vector", "tiling". These match well in search engines, but they miss context — "vector" could be a math concept, a data structure, or a vector processor.

### The Fusion Approach

We combine both:

```python
# LLM extracts semantic concepts (weight: 0.6)
llm_keywords = extract_via_llm(document)

# NLP extracts high-frequency terms (weight: 0.4)
nlp_keywords = extract_via_nlp(document)  # KeyBERT + YAKE + spaCy ensemble

# Fusion: terms appearing in both get 1.5x boost
for kw in all_keywords:
    if kw in llm_keywords and kw in nlp_keywords:
        kw.score *= 1.5
```

The LLM understands *what the paper is about*. The NLP knows *what terms appear in searchable paper titles*. Together, they produce keywords that are both semantically relevant and practically searchable.

### Handling Multi-File Projects

Theses and dissertations are multi-file LaTeX projects (`\input{chapters/introduction}`, `\include{chapters/methods}`). CiteBot recursively resolves these references and extracts keywords **per chapter with cumulative context**:

1. LLM generates a project-level summary first
2. Per-chapter extraction includes the summary + previously extracted keywords
3. This prevents duplicate keywords across chapters and maintains consistency

## Async Multi-Source Search

We search five academic databases in parallel:

| Source | Coverage | Best For |
|--------|----------|----------|
| OpenAlex | 250M+ works | Broad coverage |
| Semantic Scholar | 200M+ papers | Computer Science |
| PubMed | 36M+ citations | Biomedical |
| arXiv | 2M+ preprints | Recent STEM |
| BioRxiv | Biology preprints | Biology |

Search queries are constructed at three specificity levels:

- **Broad**: short keywords, unquoted (high recall)
- **Medium**: keyword + quoted phrase (balanced)
- **Targeted**: individual quoted phrases (high precision)

All queries execute via `asyncio.gather()` — a full search across all sources completes in 3-5 seconds.

## Relevance Scoring

Raw search results are noisy. We rank them using a composite score:

```
score = 0.50 × keyword_overlap
      + 0.15 × log(citations / 10000)
      + 0.15 × exp(-0.14 × age_years)
      + 0.20 × abstract_similarity
```

Keyword overlap dominates intentionally — it keeps results on-topic. Citation count and recency provide supporting signals. Abstract similarity catches papers that are relevant but use different terminology.

### Deduplication

The same paper appears in multiple databases with slightly different metadata. We deduplicate in two passes:

1. **DOI matching**: exact (authoritative)
2. **Fuzzy title matching**: RapidFuzz with 90% threshold (catches formatting differences)

## BibTeX Generation

For each selected paper, we fetch BibTeX via DOI content negotiation:

```python
# Preferred: authoritative BibTeX from the publisher
response = httpx.get(
    f"https://doi.org/{doi}",
    headers={"Accept": "application/x-bibtex"}
)

# Fallback: generate from metadata
if response.status_code != 200:
    bibtex = generate_from_metadata(paper)
```

Content negotiation produces cleaner BibTeX than any metadata-based generation, since the publisher's own formatting is used.

## Design Decisions

### Immutability Everywhere

All data types are frozen dataclasses. No mutations. Updates use `dataclasses.replace()`:

```python
@dataclass(frozen=True)
class ScoredPaper:
    title: str
    score: float
    # ...

# Update: creates new object, original unchanged
updated = replace(paper, score=0.95)
```

This prevents entire classes of bugs — especially in async code where multiple coroutines process the same paper objects.

### Graceful Degradation

Every component has a fallback:
- LLM unavailable → NLP-only extraction
- One search source fails → others continue
- DOI fetch fails → metadata-based BibTeX
- BibTeX validation fails → log warning, include anyway

The tool should always produce *something useful*, even in degraded conditions.

### Never Overwrite Original Files

Citation insertion writes `.cited.tex` files — never modifying the original `.tex`. Multi-file projects get per-chapter `.cited.tex` files with updated `\input`/`\include` paths.

## Usage

```bash
# Basic: 30 references
citebot paper.tex -n 30 -o refs.bib

# Thesis: 100 references, CS-focused sources
citebot thesis/main.tex -n 100 --sources s2,openalex,arxiv

# With citation insertion
citebot paper.tex -n 50 -o refs.bib --insert-cites
```

## What I Learned

1. **Search quality > search quantity** — 10 well-targeted queries beat 100 broad ones
2. **LLM + NLP fusion is better than either alone** — they complement each other's weaknesses
3. **Immutability pays for itself** — especially in async code, frozen dataclasses eliminate an entire class of bugs
4. **Content negotiation is underused** — `Accept: application/x-bibtex` on DOI URLs gives you publisher-quality BibTeX for free
5. **Per-chapter context matters for large documents** — treating a thesis as a single document produces unfocused keywords

CiteBot is open source at [github.com/Hayden727/CiteBot](https://github.com/Hayden727/CiteBot). Contributions welcome.
