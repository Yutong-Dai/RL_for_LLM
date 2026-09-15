# Xcode 26 changed lipo's CLI, which breaks TeX Live's universal biber binary.
# Use the project wrapper, which falls back to a single-arch copy when needed.
$biber = 'scripts/run-biber %O %S';
