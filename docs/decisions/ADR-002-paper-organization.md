# ADR-002: Separate drafts and final submission

Date: 2026-10-09. Status: accepted for repository organization, explicitly requested by the user.

Keep each working paper in paper/drafts/<name>/ with an independent LaTeX entry point and bibliography. Preserve the supplied tanh draft and the earlier adaptive-handover scaffold separately. Keep paper/final/ for a complete copy of the selected submission project; it is a placeholder until selection.

This supersedes only the flat paper layout used during initial setup. It does not change the academic direction, validate draft claims, or record adviser approval. The final submission gate checks final/ and allows unfinished historical drafts to remain in drafts/.
