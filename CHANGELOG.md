# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Multi-step pipeline UI with dark theme and step-based sidebar navigation
- Chapter selection with checkboxes and select-all during analysis
- Per-chapter analysis status tracking with live progress and ETA estimates
- Character table with dark-themed select inputs and confidence bars
- Corrections store for alias merges, gender overrides, and voice overrides
- SQLite-based persistent audiobook store (scaffolding)
- Shared TypeScript types module
- Worker call helper for Tauri invoke abstraction
- Tauri protocol-asset feature for temp file playback
- Custom scrollbar styling and responsive breakpoints
- License (MIT) and rights gate for copyright compliance
- Chapter audio generation with per-character voice synthesis (Parler TTS)
- Pluggable TTS backend interface
- Dialogue-aware audiobook script construction
- LLM analysis adapter for OpenAI-compatible APIs
- Dialogue and narration segmentation
- Chapter detection from extracted text
- Scanned PDF detection for OCR fallback
- Text extraction from EPUB and PDF
- Python worker CLI foundation
- Worker JSON protocol definitions
- Audiobook script IR schema

### Fixed

- Dialogue segmenter: handle inverted speech tags (e.g. `cried his wife`)
- Dialogue segmenter: handle `Mrs.` title prefixes in speaker detection
- Smart quote matching for Gutenberg EPUBs
- Gutenberg text normalization for drop-caps and orphaned quotes
- Float16 → float32 conversion for MPS TTS compatibility

[Unreleased]: https://github.com/zhhs-git/Voiceover/compare/v0.1.0...HEAD
