---
layout: project
title: "CiteBot"
title_en: "CiteBot — Intelligent Citation Assistant"
date: 2026-02-10
status: "开发中"
status_en: "Active"
excerpt: "智能学术引用工具"
excerpt_en: "An intelligent LaTeX citation assistant that automates reference discovery and BibTeX generation using LLM + NLP fusion"
---

# CiteBot — Intelligent Citation Assistant

An open-source tool that automates academic citation workflows. Give it a `.tex` file, and it produces a complete `references.bib` with relevant, well-formatted references.

## How It Works

1. **Parse** LaTeX documents (supports multi-file thesis projects)
2. **Extract** keywords using LLM semantic analysis + NLP ensemble (KeyBERT, YAKE, spaCy)
3. **Search** 5 academic databases in parallel (OpenAlex, Semantic Scholar, PubMed, arXiv, BioRxiv)
4. **Rank** results using composite scoring (keyword overlap, citations, recency, abstract similarity)
5. **Generate** BibTeX via DOI content negotiation
6. **Insert** `\cite{}` commands with fuzzy title matching (optional)

## Key Features

- **LLM + NLP fusion**: Combines semantic understanding with statistical term extraction; terms appearing in both get a 1.5x relevance boost
- **Multi-file support**: Recursive `\input`/`\include` resolution with per-chapter context-aware extraction
- **Async search**: All database queries execute in parallel via asyncio
- **Immutable data pipeline**: Frozen dataclasses throughout, preventing mutation bugs in async code
- **Graceful degradation**: Every component has a fallback — LLM→NLP, DOI→metadata, source failures skip gracefully

## Links

- [GitHub Repository](https://github.com/Hayden727/CiteBot)
