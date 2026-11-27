#!/bin/bash
# Run a command in the sandbox as the agent user (never as root: root could change the
# firewall). With no command, a shell. The container is started if it is not running.
#
#   sandbox/agent.sh                      # bash in the project
#   sandbox/agent.sh claude               # Claude Code, interactive, in the sandbox
#   sandbox/agent.sh up | down | logs     # the container itself
#   sandbox/agent.sh python3 scripts/claude_loop.py --goal ... --target ...
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$here/.env" ] || { echo "agent: run sandbox/setup.sh first" >&2; exit 1; }
compose=(docker compose -f "$here/compose.yml")
case "${1:-}" in
	up) exec "${compose[@]}" up -d ;;
	down) exec "${compose[@]}" down ;;
	logs) exec "${compose[@]}" logs claude ;;
esac
if ! "${compose[@]}" ps --status running -q claude | grep -q .; then
	"${compose[@]}" up -d
	sleep 3
fi
tty=()
[ -t 0 ] && [ -t 1 ] || tty=(-T)
[ $# -gt 0 ] || set -- bash
exec "${compose[@]}" exec "${tty[@]}" -u agent claude "$@"
