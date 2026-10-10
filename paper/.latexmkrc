# Run `latexmk` in this folder to build every paper into build/.
# Overleaf ignores this file; VS Code LaTeX Workshop and CI use it.
$pdf_mode = 1;
$out_dir = 'build';
@default_files = ('main.tex', 'drafts/*.tex');
