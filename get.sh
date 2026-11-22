#!/usr/bin/env bash
# The local Ada port uses the setup menu from the same checkout.
set -euo pipefail
if [[ ! -f "${BASH_SOURCE[0]:-}" ]]; then
  echo 'Run bash setup.sh from the local RTX 4090 checkout.' >&2
  exit 1
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$root/setup.sh" "$@"
