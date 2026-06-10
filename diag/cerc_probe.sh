#!/usr/bin/env bash
# CERC connectivity probe — run from BOTH a US vantage (GitHub Actions) and an
# Indian vantage (your laptop) and compare. It separates three failure modes:
#   geofence  -> connects but 403/blocked page, OR connection refused/reset/timeout
#   legacy TLS -> fails on default TLS but succeeds with --tls-max 1.2 + SECLEVEL=1
#   fine       -> 200 on default TLS
#
# Optional: set CERC_PROXY=http://user:pass@host:port to also test via a proxy.

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
HOSTS=("cercind.gov.in" "www.cercind.gov.in")
PATH_TO_GET="recent_orders.html"
PROXY_ARG=""
[ -n "$CERC_PROXY" ] && PROXY_ARG="--proxy $CERC_PROXY" && echo "NOTE: using CERC_PROXY (value hidden)"

curl_exit_hint() {
  case "$1" in
    0) echo "ok" ;; 6) echo "DNS: cannot resolve host" ;;
    7) echo "connect refused/blocked (possible geofence at network layer)" ;;
    28) echo "timeout (possible geofence/drop)" ;;
    35) echo "TLS handshake error (legacy-TLS or geofence RST)" ;;
    56) echo "connection reset mid-stream (possible geofence RST)" ;;
    *) echo "curl exit $1" ;;
  esac
}

probe() {  # $1=host  $2=label  $3...=extra curl args
  local host="$1" label="$2"; shift 2
  local url="https://${host}/${PATH_TO_GET}"
  local out code ip exit
  out=$(curl -sS -o /dev/null \
        -w 'code=%{http_code} ip=%{remote_ip} t=%{time_total}s' \
        --connect-timeout 15 --max-time 45 \
        -A "$UA" $PROXY_ARG "$@" "$url" 2>/dev/null)
  exit=$?
  if [ $exit -ne 0 ]; then
    printf '  %-26s FAIL  (%s)\n' "$label" "$(curl_exit_hint $exit)"
  else
    printf '  %-26s %s\n' "$label" "$out"
  fi
}

echo "================ CERC PROBE  ($(date -u +%FT%TZ)) ================"
echo "Public egress IP: $(curl -sS --max-time 15 https://api.ipify.org 2>/dev/null || echo '??')"
echo

for h in "${HOSTS[@]}"; do
  echo "---- $h ----"
  probe "$h" "default-TLS (like requests)"
  probe "$h" "TLS1.2+SECLEVEL=1 (curl)" -k --tls-max 1.2 --ciphers 'DEFAULT@SECLEVEL=1'
  # Pull the first ~200 bytes of body to catch a 'blocked'/'not available in your region' page
  body=$(curl -sS -k --tls-max 1.2 --ciphers 'DEFAULT@SECLEVEL=1' \
         --connect-timeout 15 --max-time 45 -A "$UA" $PROXY_ARG \
         "https://${h}/${PATH_TO_GET}" 2>/dev/null | tr -d '\r\n\t' | head -c 200)
  [ -n "$body" ] && echo "  body[0:200]: ${body}"
  echo
done
echo "Read: 200 on default-TLS = fine. 200 only with TLS1.2 flags = legacy-TLS."
echo "      403 / refused / timeout / reset on both = geofence."
