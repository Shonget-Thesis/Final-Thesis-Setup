# ADR-004: One manuscript, drafts kept in git history

Date: 2026-10-24. Status: accepted, explicitly requested by Mark Vincent D. Limpahan (team member). Supersedes the draft parts of ADR-002 and ADR-003.

Context: the team's proposal now lives in paper/main.tex and paper/chapters/ in the CITC format (PAPER-008). The standalone drafts paper/drafts/tanh-velocity-blending.tex and paper/drafts/adaptive-handover.tex were no longer edited, and the team removed them to keep the paper folder to a single project.

Choice: paper/ holds one LaTeX project, paper/main.tex. Fill-in fields for the preliminary pages are in paper/frontmatter.tex. The drafts are removed from the working tree; both remain in git history (last present at commit 2e51c49), and the supplied tanh source with its hash record stays in paper/sources/.

Alternatives: keep the drafts in paper/drafts/ (previous rule) or move them to an archive folder. The team chose git history as the archive.

Consequences: latexmk, CI and the workspace check build and check only paper/main.tex. Text reused from an old draft should cite the source commit. This is a team organization decision, not adviser approval.
