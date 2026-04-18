#!/bin/bash
set -e

# Benchmark script for kb compilation performance
# Measures three scenarios:
# 1. Full rebuild (clean)
# 2. No-op incremental (no changes)
# 3. One-source-changed incremental

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
KB_BIN="$PROJECT_ROOT/target/release/kb"
CORPUS_PATH="$PROJECT_ROOT/crates/kb-cli/tests/fixtures/corpus-tiny"
RUNS=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building kb binary in release mode...${NC}"
maw exec bn-3a6 -- cargo build --release --quiet

echo -e "${YELLOW}Creating temporary KB directory...${NC}"
TEMP_KB=$(mktemp -d)
trap "rm -rf $TEMP_KB" EXIT

echo "Using temporary KB: $TEMP_KB"

# Initialize KB
echo -e "${YELLOW}Initializing KB...${NC}"
"$KB_BIN" --root "$TEMP_KB" init > /dev/null

# Create corpus files in wiki/sources
echo -e "${YELLOW}Creating source pages from test corpus...${NC}"
for file in "$CORPUS_PATH"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        slug="${filename%.md}"
        title=$(head -1 "$file" | sed 's/^# //' | sed 's/^## //')
        [ -z "$title" ] && title="$slug"

        # Read the file content
        content=$(cat "$file")

        # Write as wiki source page
        cat > "$TEMP_KB/wiki/sources/$slug.md" << EOF
---
id: wiki-source-$slug
type: source
title: $title
---

# Source
<!-- kb:begin id=title -->
$title
<!-- kb:end id=title -->

## Summary
<!-- kb:begin id=summary -->
$content
<!-- kb:end id=summary -->
EOF
    fi
done

# Also add some concept pages for a more realistic KB
echo -e "${YELLOW}Creating concept pages...${NC}"
cat > "$TEMP_KB/wiki/concepts/test-concept.md" << 'EOF'
---
id: concept:test-concept
name: Test Concept
aliases:
  - test
  - sample
---

# Test Concept

This is a sample concept page created for benchmarking.
EOF

# Function to run a single compile and return timing in milliseconds
run_compile() {
    local start_time=$(date +%s%N)
    "$KB_BIN" --root "$TEMP_KB" compile > /dev/null 2>&1
    local end_time=$(date +%s%N)
    local elapsed=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
    echo $elapsed
}

# Function to calculate percentile
percentile() {
    local p=$1
    shift
    local sorted=($(printf '%s\n' "$@" | sort -n))
    local index=$(( (${#sorted[@]} * p / 100) ))
    if [ $index -lt 0 ]; then index=0; fi
    if [ $index -ge ${#sorted[@]} ]; then index=$(( ${#sorted[@]} - 1 )); fi
    echo ${sorted[$index]}
}

echo ""
echo -e "${YELLOW}Scenario 1: Full rebuild (clean)${NC}"
declare -a full_rebuild_times
for i in $(seq 1 $RUNS); do
    echo -n "  Run $i/$RUNS... "
    # Clean the compiled artifacts to force a full rebuild
    rm -rf "$TEMP_KB/outputs" "$TEMP_KB/state/indexes" "$TEMP_KB/normalized"
    # Reinitialize the normalized directory
    mkdir -p "$TEMP_KB/normalized"
    time_ms=$(run_compile)
    full_rebuild_times+=($time_ms)
    echo "${time_ms}ms"
done

echo ""
echo -e "${YELLOW}Scenario 2: No-op incremental (no changes)${NC}"
declare -a noop_times
for i in $(seq 1 $RUNS); do
    echo -n "  Run $i/$RUNS... "
    time_ms=$(run_compile)
    noop_times+=($time_ms)
    echo "${time_ms}ms"
done

echo ""
echo -e "${YELLOW}Scenario 3: One-source-changed incremental${NC}"
declare -a changed_times
for i in $(seq 1 $RUNS); do
    echo -n "  Run $i/$RUNS... "
    # Modify the first source file (add a comment)
    echo "" >> "$TEMP_KB/wiki/sources/basic.md"
    echo "<!-- Modified in benchmark run $i -->" >> "$TEMP_KB/wiki/sources/basic.md"
    time_ms=$(run_compile)
    changed_times+=($time_ms)
    echo "${time_ms}ms"
done

echo ""
echo -e "${GREEN}=== Benchmark Results ===${NC}"

# Calculate statistics
full_p50=$(percentile 50 "${full_rebuild_times[@]}")
full_p95=$(percentile 95 "${full_rebuild_times[@]}")
full_min=$(printf '%s\n' "${full_rebuild_times[@]}" | sort -n | head -1)
full_max=$(printf '%s\n' "${full_rebuild_times[@]}" | sort -n | tail -1)

noop_p50=$(percentile 50 "${noop_times[@]}")
noop_p95=$(percentile 95 "${noop_times[@]}")
noop_min=$(printf '%s\n' "${noop_times[@]}" | sort -n | head -1)
noop_max=$(printf '%s\n' "${noop_times[@]}" | sort -n | tail -1)

changed_p50=$(percentile 50 "${changed_times[@]}")
changed_p95=$(percentile 95 "${changed_times[@]}")
changed_min=$(printf '%s\n' "${changed_times[@]}" | sort -n | head -1)
changed_max=$(printf '%s\n' "${changed_times[@]}" | sort -n | tail -1)

# Calculate speedup (use floating point for better accuracy)
speedup=$(echo "scale=2; $full_p50 / $changed_p50" | bc -l 2>/dev/null || echo "N/A")

# Create output directory
mkdir -p "$PROJECT_ROOT/docs/perf"

cat > "$PROJECT_ROOT/docs/perf/baseline.md" << EOF
# KB Compilation Performance Baseline

Benchmark date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Test Setup

- Corpus: corpus-tiny (6 markdown documents + 1 concept page)
- Runs per scenario: $RUNS
- System: $(uname -s | sed 's/Linux/Linux/') $(uname -m)
- KB Binary: release mode

## Results

### Scenario 1: Full Rebuild (Clean)

Starting from cleaned state (removed outputs/ and compiled artifacts), measure the time to compile the entire knowledge base from scratch.

- p50: ${full_p50}ms
- p95: ${full_p95}ms
- min: ${full_min}ms
- max: ${full_max}ms

### Scenario 2: No-Op Incremental

Running compile without making any changes to the sources. This tests the cache/hash verification overhead and validates that the system correctly detects no changes are needed.

- p50: ${noop_p50}ms
- p95: ${noop_p95}ms
- min: ${noop_min}ms
- max: ${noop_max}ms
- **Status**: $([ $noop_p95 -lt 1000 ] && echo "✓ Under 1s target" || echo "✗ Exceeds 1s target")

### Scenario 3: One-Source-Changed Incremental

Modifying a single source file and measuring the incremental recompilation time. This tests how well the system can skip unchanged sources.

- p50: ${changed_p50}ms
- p95: ${changed_p95}ms
- min: ${changed_min}ms
- max: ${changed_max}ms
- Speedup vs full rebuild (p50): ${speedup}x faster

## Acceptance Criteria

✓ Three measured scenarios
- $([ $noop_p95 -lt 1000 ] && echo "✓" || echo "✗") Incremental no-op is under 1s (p95: ${noop_p95}ms)
- $([ $(echo "$speedup > 1.0" | bc -l 2>/dev/null || echo "0") == "1" ] && echo "✓" || echo "✗") One-source-changed is proportionally faster than full rebuild (${speedup}x)

## Notes

- All times are in milliseconds
- Percentiles help account for system variance and outliers
- p50 represents the median time
- p95 represents the 95th percentile (worst 5% of runs)
- The test corpus is small (corpus-tiny); production performance with larger corpora may differ significantly
- Full rebuilds force removal of compiled artifacts to ensure they're not using any caches

## Performance Interpretation

- **No-op incremental under 1s**: Ensures the user can keep the tool in their loop without friction
- **Incremental faster than full rebuild**: Validates the incremental compilation strategy
- **p95 < 2x p50**: Indicates the system is stable without significant outliers
EOF

echo ""
echo "Full rebuild    p50: ${full_p50}ms, p95: ${full_p95}ms"
echo "No-op           p50: ${noop_p50}ms, p95: ${noop_p95}ms"
echo "One-file change p50: ${changed_p50}ms, p95: ${changed_p95}ms (${speedup}x faster)"
echo ""
echo -e "${GREEN}Results saved to docs/perf/baseline.md${NC}"
