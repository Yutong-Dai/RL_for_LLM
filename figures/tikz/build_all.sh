#!/usr/bin/env bash
# Usage:
#   From the repository root:
#     ./figures/tikz/build_all.sh
#   Or from this directory:
#     ./build_all.sh
#
# The script compiles all TikZ .tex files in this directory and writes PDFs to
# ../, keeping generated PDFs out of figures/tikz/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

shopt -s nullglob
tex_files=("${SCRIPT_DIR}"/*.tex)

if [ "${#tex_files[@]}" -eq 0 ]; then
  echo "No TikZ .tex files found in ${SCRIPT_DIR}."
  rm -f "${SCRIPT_DIR}"/*.pdf
  exit 0
fi

for tex_file in "${tex_files[@]}"; do
  base_name="$(basename "${tex_file}" .tex)"
  echo "Building ${base_name}.pdf"

  pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="${OUTPUT_DIR}" \
    "${tex_file}"

  rm -f \
    "${OUTPUT_DIR}/${base_name}.aux" \
    "${OUTPUT_DIR}/${base_name}.log" \
    "${OUTPUT_DIR}/${base_name}.out"
done

# LaTeX Workshop may compile focused TikZ files in this directory. Keep PDFs one
# level above so main.tex can include figures/<name>.pdf consistently.
rm -f "${SCRIPT_DIR}"/*.pdf

echo "Built ${#tex_files[@]} TikZ figure(s) into ${OUTPUT_DIR}."
