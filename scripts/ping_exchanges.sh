#!/usr/bin/env bash
set -euo pipefail

# Edit these hosts/paths при необходимости.
# Bybit
BYBIT_API_HOST="api.bybit.com"
BYBIT_API_PATH="/v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=1"
BYBIT_WS_HOST="stream.bybit.com"

# Gate.io
GATE_API_HOST="api.gateio.ws"
GATE_API_PATH="/api/v4/spot/currencies"
GATE_WS_HOST="stream.gateio.ws"

# Bitunix (проверьте домены, при необходимости поправьте)
BITUNIX_API_HOST="api.bitunix.com"
BITUNIX_API_PATH="/api/v1/market/tickers"
BITUNIX_WS_HOST="ws.bitunix.com"

NUM_PINGS=3
HTTP_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --http-only)
      HTTP_ONLY=1
      ;;
  esac
done

cyan()  { printf "\033[36m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*"; }

test_icmp() {
  local host="$1"
  cyan "[ICMP] ping $host ($NUM_PINGS packets)"
  if command -v ping >/dev/null 2>&1; then
    # Skip on request
    if [ "$HTTP_ONLY" = "1" ]; then
      echo "  skipped (--http-only)"
      return 0
    fi
    # Detect Linux/Mac vs Windows ping syntax
    if ping -c 1 127.0.0.1 >/dev/null 2>&1; then
      # Unix-like: -n numeric, -c count
      ping -n -c "$NUM_PINGS" "$host" | sed 's/^/  /'
    else
      # Windows ping (Git Bash uses Windows ping.exe): -n count
      # Перекодируем OEM CP866 -> UTF-8, если iconv доступен
      if command -v iconv >/dev/null 2>&1; then
        ping -n "$NUM_PINGS" "$host" | iconv -f CP866 -t UTF-8 2>/dev/null | sed 's/^/  /'
      else
        ping -n "$NUM_PINGS" "$host" | sed 's/^/  /'
      fi
    fi
  else
    red "ping not found"
  fi
}

test_http() {
  local host="$1"; shift
  local path="${1:-/}"
  local url="https://$host$path"
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
  local api_path="$1"; shift
  local ws_host="$1"; shift
  echo
  green "===== $name ====="
  test_icmp "$api_host"
  test_http "$api_host" "$api_path"
  test_icmp "$ws_host"
  # Для WS достаточно TLS‑handshake/TTFB на корне хоста
  test_http "$ws_host" "/"
}

run_suite "Bybit"   "$BYBIT_API_HOST"   "$BYBIT_API_PATH"   "$BYBIT_WS_HOST"
run_suite "Gate.io" "$GATE_API_HOST"    "$GATE_API_PATH"    "$GATE_WS_HOST"
run_suite "Bitunix" "$BITUNIX_API_HOST" "$BITUNIX_API_PATH" "$BITUNIX_WS_HOST"

echo
green "Done. Меняйте HOST/PATH вверху при необходимости. Интерпретация:"
echo "- ICMP avg RTT — оценка сетевой задержки (может быть заблокирован фаерволом)."
echo "- time_connect — TCP‑handshake до 443 (примерно равен RTT)."
echo "- time_appconnect — TLS‑handshake; платится 1 раз на подключение."
echo "- time_starttransfer — TTFB для HTTP; для WS важнее RTT/TLS, не TTFB."


