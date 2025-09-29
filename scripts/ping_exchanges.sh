#!/usr/bin/env bash
set -euo pipefail

# Edit these hosts if нужно. Добавлены типичные публичные домены API/WS.
BYBIT_API="api.bybit.com"
BYBIT_WS="stream.bybit.com"

GATE_API="api.gateio.ws"
GATE_WS="stream.gateio.ws"

# Для Bitunix проверьте правильные домены и поправьте при необходимости
BITUNIX_API="api.bitunix.com"
BITUNIX_WS="ws.bitunix.com"

NUM_PINGS=3

cyan()  { printf "\033[36m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

test_icmp() {
  local host="$1"
  cyan "[ICMP] ping $host ($NUM_PINGS packets)"
  if command -v ping >/dev/null 2>&1; then
    # -n numeric, -c count
    ping -n -c "$NUM_PINGS" "$host" | sed 's/^/  /'
  else
    red "ping not found"
  fi
}

test_http() {
  local host="$1"
  local url="https://$host/"
  cyan "[HTTP/TLS] curl $url"
  if command -v curl >/dev/null 2>&1; then
    curl -s -o /dev/null -w "  time_namelookup: %{time_namelookup}s\n  time_connect:    %{time_connect}s\n  time_appconnect: %{time_appconnect}s (TLS)\n  time_starttransfer: %{time_starttransfer}s\n  time_total:      %{time_total}s\n  http_code:       %{http_code}\n" "$url" || true
  else
    red "curl not found"
  fi
}

run_suite() {
  local name="$1"; shift
  local api_host="$1"; shift
  local ws_host="$1"; shift
  echo
  green "===== $name ====="
  test_icmp "$api_host"
  test_http "$api_host"
  test_icmp "$ws_host"
  test_http "$ws_host"
}

run_suite "Bybit"   "$BYBIT_API"   "$BYBIT_WS"
run_suite "Gate.io" "$GATE_API"    "$GATE_WS"
run_suite "Bitunix" "$BITUNIX_API" "$BITUNIX_WS"

echo
green "Done. При необходимости отредактируйте hosts в scripts/ping_exchanges.sh"


