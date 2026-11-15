#!/usr/bin/env bash
# Install the wheel from dist/ into a fresh venv and run pytest from the repository root.
#
#   .github/scripts/pytest-wheel.sh PYTHON [--lowest] [--with REQ]... [PYTEST ARGS]...
#
# --lowest installs the lowest versions that pyproject.toml allows for the direct dependencies
# (numpy, Pillow). --with adds a requirement, for example a pinned Pillow.
set -euo pipefail

python=$1
shift
resolution=highest
extra=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --lowest) resolution=lowest-direct; shift ;;
    --with) extra+=("$2"); shift 2 ;;
    *) break ;;
  esac
done

wheels=(dist/*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected exactly one wheel in dist/, found: ${wheels[*]}" >&2
  exit 1
fi

venv=$(mktemp -d)/venv
uv venv --python "$python" "$venv"
# numpy and pillow are listed so that --resolution lowest-direct applies to them.
uv pip install --python "$venv" --resolution "$resolution" \
  "${wheels[0]}" numpy pillow 'pytest>=8' ${extra[@]+"${extra[@]}"}
uv pip list --python "$venv"

if [[ -x $venv/bin/python ]]; then py=$venv/bin/python; else py=$venv/Scripts/python.exe; fi
# The installed wheel, not python/manyfaced, must be imported (python/ is not on sys.path).
"$py" -c 'import manyfaced, sys; print("manyfaced from", manyfaced.__file__); sys.exit("site-packages" not in manyfaced.__file__)'
exec "$py" -m pytest "$@"
