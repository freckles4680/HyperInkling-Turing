#!/bin/bash
# Outbound allowlist for the sandbox, after the firewall of Claude Code's reference dev
# container: loopback (the test servers), DNS, and HTTPS to the hosts below. Everything else,
# GitHub and Hugging Face included, is refused. Nothing listens for the outside (compose.yml
# publishes no port).
#
# ALLOW_DOMAINS (sandbox/.env) adds hosts, e.g. "pypi.org files.pythonhosted.org" for a pip
# install. The names are resolved once, at container start: restart the container after a
# change, or when a host's addresses moved.
set -euo pipefail

domains="api.anthropic.com claude.ai claude.com console.anthropic.com platform.claude.com ${ALLOW_DOMAINS:-}"

iptables -F INPUT
iptables -F OUTPUT
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
# DNS (Docker's resolver forwards from inside the container). A name lookup can still carry
# a few bytes out: accepted here, as in the reference container.
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

for domain in $domains; do
	ips=$(getent ahostsv4 "$domain" | awk '{print $1}' | sort -u || true)
	if [ -z "$ips" ]; then
		echo "firewall: cannot resolve $domain, left out" >&2
		continue
	fi
	for ip in $ips; do
		iptables -A OUTPUT -p tcp -d "$ip" --dport 443 -j ACCEPT
	done
done
# Refuse (not drop) the rest, so that a blocked call fails at once instead of timing out.
iptables -A OUTPUT -j REJECT --reject-with icmp-admin-prohibited

if command -v ip6tables > /dev/null && ip6tables -L > /dev/null 2>&1; then
	ip6tables -F INPUT
	ip6tables -F OUTPUT
	ip6tables -P INPUT DROP
	ip6tables -P FORWARD DROP
	ip6tables -P OUTPUT DROP
	ip6tables -A INPUT -i lo -j ACCEPT
	ip6tables -A OUTPUT -o lo -j ACCEPT
fi

# Check both ways: an allowed host answers, any other does not.
if curl -s --max-time 5 -o /dev/null https://example.com; then
	echo "firewall: example.com is reachable, the allowlist is not working" >&2
	exit 1
fi
code=$(curl -s --max-time 10 -o /dev/null -w '%{http_code}' https://api.anthropic.com || true)
if [ "$code" = 000 ]; then
	echo "firewall: api.anthropic.com is not reachable" >&2
	exit 1
fi
echo "firewall: outbound limited to loopback, DNS and https://{$(echo "$domains" | xargs | tr ' ' ',')}"
