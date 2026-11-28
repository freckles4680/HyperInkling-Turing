#!/bin/bash
# Serve every model in scripts/models.toml on one address, loading the one each request
# names (Inkling 2 Mini for chat, Clef for decisions). Only one fits on the 2080 Ti, so a
# request for the other one waits for the switch: Inkling takes a minute or two to boot,
# Clef about as long to load (docs/clef.md).
#
#   scripts/serve-switch.sh                       # foreground; run it inside tmux
#   PRELOAD=inkling-2-mini scripts/serve-switch.sh   # boot Inkling now, not on the first request
#   API_KEY=secret BIND=127.0.0.1 scripts/serve-switch.sh
#
# Point Open WebUI at  http://<this machine>:8081/v1  (with the speed line) or :8080/v1,
# as with scripts/serve.sh: the addresses do not change. Each model's own settings stay
# in its script (scripts/serve.sh, scripts/clef.sh); models.toml only says how to start it.
#
# Everything, the switches and what each server prints, goes into one file to share:
#   .local/model-switch.log
# Which model is loaded:   curl -s http://127.0.0.1:8080/switch/status
# Free the GPU:            curl -s -X POST http://127.0.0.1:8080/switch/unload
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 0.0.0.0 = the LAN, 127.0.0.1 = this machine only.
export BIND=${BIND:-0.0.0.0}
# The public port; the models themselves listen on 127.0.0.1 (ports in models.toml).
export PORT=${PORT:-8080}
# Checked here for every model ("Authorization: Bearer <key>"); empty = no key.
export API_KEY=${API_KEY:-}
# Which models and how to start them.
CONFIG=${CONFIG:-scripts/models.toml}
# A model to start at once (empty = on the first request that names one).
export PRELOAD=${PRELOAD:-}
# The speed line under each chat answer (scripts/speed_proxy.py); empty = off.
SPEED_PORT=${SPEED_PORT-8081}
export SWITCH_LOG=${SWITCH_LOG:-.local/model-switch.log}

mkdir -p "$(dirname "$SWITCH_LOG")"
if [ -f "$SWITCH_LOG" ]; then mv -f "$SWITCH_LOG" "$SWITCH_LOG.1"; fi
if ss -ltn 2> /dev/null | awk '{print $4}' | grep -qE "[:.]$PORT$"; then
	echo "serve-switch: port $PORT is already in use (scripts/serve.sh still running?)" >&2
	exit 1
fi
if [ "$BIND" = 0.0.0.0 ]; then
	address=$(hostname -I 2> /dev/null | awk '{print $1}')
	local_address=127.0.0.1
else
	address=$BIND
	local_address=$BIND
fi
echo "serve-switch: http://${address:-<this machine>}:$PORT/v1" \
	"$([ -n "$API_KEY" ] && echo '(API key required)' || echo '(no API key)')"
if [ -n "$SPEED_PORT" ]; then
	echo "serve-switch: with the speed line: http://${address:-<this machine>}:$SPEED_PORT/v1"
	python3 scripts/speed_proxy.py --bind "$BIND" --port "$SPEED_PORT" \
		--upstream "http://$local_address:$PORT" &
	proxy=$!
	trap 'kill "$proxy" 2> /dev/null || true' EXIT
fi
python3 scripts/model_switch.py --config "$CONFIG" --bind "$BIND" --port "$PORT" \
	--log "$SWITCH_LOG"
