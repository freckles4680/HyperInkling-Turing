#!/bin/bash
# Container entrypoint: firewall as root, then the command as the agent user with every
# capability dropped (it cannot change the firewall afterwards).
set -euo pipefail
/usr/local/sbin/init-firewall.sh
exec setpriv --reuid=agent --regid="$(id -g agent)" --init-groups \
	--inh-caps=-all --bounding-set=-all --no-new-privs -- "$@"
