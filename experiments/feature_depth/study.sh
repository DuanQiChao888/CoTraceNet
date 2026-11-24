#!/usr/bin/env bash
# Run the registered capacity comparison. Build the caches and baseline first.
set -euo pipefail
: "${FEATURES:?Set FEATURES to the joint descriptor cache root}"
: "${POLICY:?Set POLICY to the trained binocular policy directory}"
: "${GRIDS:?Set GRIDS to the spatial grid cache root}"
: "${STUDY:?Set STUDY to the experiment output directory}"
PYTHON=${PYTHON:-python}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

run() {
  local module=$1
  shift
  "$PYTHON" -m "cotracenet.feature_depth.$module" "$@"
}

candidate() {
  local name=$1 phase=$2 source=$3 seed=$4 output=$5 target
  case "$name" in
    hierarchy|linear)
      run "$name" "$phase" --train "$source" --root "$FEATURES" --policy "$POLICY" --out "$output" --seed "$seed"
      ;;
    resmlp)
      if [[ $phase == fit ]]; then
        local models=(resmlp resmlp_shallow)
        if [[ $seed == 29 ]]; then models=(resmlp); fi
        run resmlp fit --models "${models[@]}" --train "$source" --features "$FEATURES" --archive "$POLICY" --out "$output" --seed "$seed"
      else
        for target in WOW HILL; do
          run resmlp test --train "$source" --target "$target" --features "$FEATURES" --archive "$POLICY" --out "$output"
        done
      fi
      ;;
    spatial)
      if [[ $phase == fit ]]; then
        run spatial plan --source "$source" --cache "$GRIDS" --policy "$POLICY" --out "$output" --seed "$seed"
      fi
      run spatial "$phase" --source "$source" --cache "$GRIDS" --policy "$POLICY" --out "$output" --seed "$seed"
      ;;
  esac
}

case "${1:-}" in
  fit)
    mkdir "$STUDY"
    for source in WOW HILL; do
      for name in hierarchy resmlp spatial; do
        candidate "$name" fit "$source" 73 "$STUDY/$name"
      done
      candidate linear fit "$source" 73 "$STUDY/hierarchy_control"
    done
    run compare lock --root "$STUDY"
    winner=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["repeat_candidate"])' "$STUDY/selection-lock.json")
    for source in WOW HILL; do
      candidate "$winner" fit "$source" 29 "$STUDY/repeat29/$winner"
    done
    run compare lock --root "$STUDY/repeat29" --parent-lock "$STUDY/selection-lock.json"
    (cd "$STUDY" && sha256sum hierarchy_control/*/unified-models.json hierarchy_control/*/unified-val.npz > control-before-test.sha256)
    ;;
  test)
    (cd "$STUDY" && sha256sum -c control-before-test.sha256)
    winner=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["repeat_candidate"])' "$STUDY/selection-lock.json")
    for source in WOW HILL; do
      for name in hierarchy resmlp spatial; do
        candidate "$name" test "$source" 73 "$STUDY/$name"
      done
      candidate linear test "$source" 73 "$STUDY/hierarchy_control"
      run ablate --train "$source" --split test --root "$FEATURES" --out "$STUDY/hierarchy"
      candidate "$winner" test "$source" 29 "$STUDY/repeat29/$winner"
      if [[ $winner == hierarchy ]]; then
        run ablate --train "$source" --split test --root "$FEATURES" --out "$STUDY/repeat29/hierarchy"
      fi
    done
    run compare test --root "$STUDY" --archive "$POLICY"
    run compare test --root "$STUDY/repeat29" --parent-lock "$STUDY/selection-lock.json" --archive "$POLICY"
    run linear_comparison "$STUDY"
    if [[ $winner == hierarchy ]]; then
      run hierarchy_export --root "$STUDY" --output "$STUDY/hierarchy-details.csv"
    fi
    ;;
  *) echo "Usage: bash experiments/feature_depth/study.sh {fit|test}" >&2; exit 2 ;;
esac
