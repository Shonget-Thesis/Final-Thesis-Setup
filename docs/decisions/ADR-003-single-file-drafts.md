# ADR-003: Single-file drafts, chapter-based final

Date: 2026-10-09. Status: accepted, explicitly requested by the user.

Supersedes ADR-002's draft folder layout. Each draft is paper/drafts/<name>.tex with inline metadata, text, and any bibliography. Final writing uses paper/final/main.tex, metadata.tex, references.bib, and chapters/*.tex. Source provenance stays in paper/sources/. Academic approval and content status remain unchanged.
