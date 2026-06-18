#!/usr/bin/env python3
"""
Railway MCP Connection Diagnostic Workflow
===========================================

This is a multi-"agent" diagnostic tool. Each "agent" is a focused checker that
probes ONE possible cause of the connection failure between VS Code and your
Railway-hosted MCP server. After all probe agents run, an ADVERSARIAL VERIFIER
re-checks the conclusions before anything is reported to you.

Everything is explained in plain English. No coding knowledge needed to read the
output.

HOW TO USE:
    python diagnose.py https://your-app.up.railway.app

If you don't pass a URL, it uses the one from your screenshots.
"""

import argparse
import json
from pathlib import Path
import socket
import ssl
import sys
import urllib.request
import urllib.error
from urllib.parse import urlparse
from datetime import datetime

# The URL seen in your VS Code screenshot
DEFAULT_URL = "https://trilegal-cerc-updates-production.up.railway.app"

# Common paths an MCP server might expose. Keep "/" after explicit MCP paths so
# the path scan first answers the user's practical question: "what URL goes in
# VS Code?"
CANDIDATE_PATHS = ["/mcp", "/sse", "/messages", "/health", "/"]


class Finding:
    """One result from one agent, in plain language."""
    def __init__(self, agent, severity, plain_summary, evidence, suggestion):
        self.agent = agent
        self.severity = severity  # "BLOCKER", "WARNING", "OK", "INFO"
        self.plain_summary = plain_summary
        self.evidence = evidence
        self.suggestion = suggestion
        self.verified = None  # filled in by the adversarial verifier
        self.verifier_note = ""

    def to_dict(self):
        return {
            "agent": self.agent,
            "severity": self.severity,
            "plain_summary": self.plain_summary,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "verified": self.verified,
            "verifier_note": self.verifier_note,
        }


def http_probe(url, method="GET", extra_headers=None, timeout=10, body=None):
    """Low-level helper. Returns (status_code, headers, body_text, error_text)."""
    headers = {"User-Agent": "mcp-diagnostic/1.0"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, method=method, headers=headers,
                                 data=body.encode() if body else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read(2000).decode("utf-8", "replace")
            return resp.status, dict(resp.headers), text, None
    except urllib.error.HTTPError as e:
        text = e.read(2000).decode("utf-8", "replace") if e.fp else ""
        return e.code, dict(e.headers), text, None
    except urllib.error.URLError as e:
        return None, {}, "", str(e.reason)
    except Exception as e:
        return None, {}, "", str(e)


def header_value(headers, name, default=""):
    """Case-insensitive header lookup for urllib's plain dictionaries."""
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return default


def normalize_url(raw_url):
    """
    Accept either a full URL or a bare Railway domain. VS Code needs the full
    https:// URL, but people often paste only the host while debugging.
    """
    candidate = raw_url.strip()
    if not candidate:
        raise ValueError("No URL was provided.")
    if "://" not in candidate:
        candidate = "https://" + candidate

    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Use an http:// or https:// URL.")
    if not parsed.hostname:
        raise ValueError("The URL does not contain a usable hostname.")

    return candidate, parsed


def base_url(parsed):
    return f"{parsed.scheme}://{parsed.netloc}"


def is_railway_application_not_found(status, headers, text):
    """
    Railway's edge page is identified by its body text, not merely by a 404.
    Apps commonly return their own 404 at "/" while still being reachable.
    """
    lowered = (text or "").lower()
    return status == 404 and "application not found" in lowered


def is_invalid_host_header(status, text):
    lowered = (text or "").lower()
    return status == 421 or "invalid host header" in lowered


def status_implies_path_exists(status):
    """
    MCP endpoints often reject an ordinary browser-style GET while still being
    real endpoints. These statuses are useful existence signals during a scan.
    """
    return isinstance(status, int) and (
        status < 400 or status in (400, 401, 403, 405, 406, 415, 421, 422)
    )


# ---------------------------------------------------------------------------
# PROBE AGENTS — each one investigates a single possible cause.
# ---------------------------------------------------------------------------

def agent_url_shape(parsed):
    """AGENT 0: Is the URL shaped like a Railway public MCP URL?"""
    if parsed.port and parsed.port not in (443,):
        return Finding(
            "URL Shape Checker",
            "BLOCKER",
            f"The URL includes an explicit port number (:{parsed.port}). Public Railway URLs should not include ':8000' or ':8080' in VS Code.",
            f"Parsed URL netloc is {parsed.netloc!r}.",
            f"Remove ':{parsed.port}' and use {parsed.scheme}://{parsed.hostname}{parsed.path or ''}",
        )
    if parsed.scheme != "https":
        return Finding(
            "URL Shape Checker",
            "WARNING",
            "The URL does not start with https://. Railway public domains are normally HTTPS endpoints.",
            f"Parsed scheme is {parsed.scheme!r}.",
            f"Use https://{parsed.netloc}{parsed.path or ''} in VS Code unless you are deliberately testing localhost.",
        )
    return Finding(
        "URL Shape Checker",
        "OK",
        "The URL shape looks like a normal Railway public URL: https:// host, with no public port number.",
        f"Parsed host is {parsed.hostname!r}; explicit port is {parsed.port!r}.",
        "No action needed for this part.",
    )


def agent_dns(parsed):
    """AGENT 1: Does the web address even exist / point somewhere?"""
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None)
        ips = sorted({info[4][0] for info in infos})
        shown = ", ".join(ips[:3])
        if len(ips) > 3:
            shown += f", ... ({len(ips)} total)"
        return Finding(
            "DNS Resolver",
            "OK",
            f"The web address '{host}' is real and points to a server.",
            f"Resolved {host} -> {shown}",
            "No action needed for this part.",
        )
    except Exception as e:
        return Finding(
            "DNS Resolver",
            "BLOCKER",
            f"The web address '{host}' does not point to any server. The address itself may be wrong or the public domain was never turned on in Railway.",
            f"DNS lookup failed: {e}",
            "In Railway, open your service -> Settings -> Networking and click 'Generate Domain'. Copy the EXACT domain it gives you and use that in VS Code.",
        )


def agent_tls(parsed):
    """AGENT 2: Is the secure (https) handshake working?"""
    host = parsed.hostname
    if parsed.scheme != "https":
        return Finding(
            "Secure Connection (TLS)",
            "WARNING",
            "The URL uses plain http://. Railway's public URLs should normally be used as https:// URLs in VS Code.",
            f"Scheme is {parsed.scheme!r}.",
            "Change the VS Code MCP URL to start with https:// unless you are deliberately testing a local HTTP-only server.",
        )

    port = parsed.port or 443
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                return Finding(
                    "Secure Connection (TLS)",
                    "OK",
                    "The secure padlock connection works. Encryption is not the problem.",
                    f"TLS handshake succeeded; certificate subject present: {bool(cert)}",
                    "No action needed for this part.",
                )
    except Exception as e:
        return Finding(
            "Secure Connection (TLS)",
            "WARNING",
            "The secure connection could not be completed. This is unusual for Railway and usually points to a wrong address rather than a broken certificate.",
            f"TLS error: {e}",
            "Double-check you copied the domain correctly, including no typos and no trailing slash.",
        )


def agent_app_alive(url):
    """AGENT 3: Is Railway returning 'Application not found' (edge error) vs your app answering?"""
    status, headers, text, err = http_probe(url, method="GET")
    if err:
        return Finding(
            "Is The App Reachable",
            "BLOCKER",
            "Could not reach the address at all.",
            f"Network error: {err}",
            "Confirm the service is deployed and the domain is generated in Railway.",
        )
    server = header_value(headers, "Server")
    if is_railway_application_not_found(status, headers, text):
        return Finding(
            "Is The App Reachable",
            "BLOCKER",
            "Railway itself says 'Application not found'. This means the address is NOT connected to your running app. Your app is fine — the address is the problem.",
            f"Status {status}, Server header '{server}', body contains Railway's not-found message.",
            "Most likely causes: (a) the public domain isn't attached to THIS service, (b) you're using a domain from a different/old service, or (c) the service has no domain generated. Fix in Railway -> Service -> Settings -> Networking.",
        )
    if is_invalid_host_header(status, text):
        return Finding(
            "Is The App Reachable",
            "BLOCKER",
            "The domain reaches your deployed service, but the server rejects the public Railway hostname as an invalid Host header.",
            f"Status {status}, Server header '{server}', body preview: {text[:160]!r}",
            "Add the exact Railway public domain to your app/framework's allowed-hosts or trusted-hosts setting, then redeploy. For MCP Python/ASGI servers, look for TrustedHostMiddleware, allowed_hosts, host validation, or DNS-rebinding protection settings.",
        )
    if status == 404:
        return Finding(
            "Is The App Reachable",
            "OK",
            "The domain reaches an HTTP service, and the response is NOT Railway's generic 'Application not found' page. A 404 at the root can be normal if the MCP server only serves a specific endpoint.",
            f"Status {status}, Server header '{server}', body preview: {text[:160]!r}",
            "Continue to the path scan. The root URL may simply not be an endpoint.",
        )
    return Finding(
        "Is The App Reachable",
        "OK",
        "The address reaches a live server that is NOT showing Railway's generic 'not found' page. Your app is answering.",
        f"Status {status}, Server header '{server}'.",
        "Good — the address points at something live. The issue is more likely the path or transport (see other agents).",
    )


def agent_path_scan(base):
    """AGENT 4: Which paths answer? Where does /mcp actually live?"""
    results = {}
    edge_paths = []
    invalid_host_paths = []
    for path in CANDIDATE_PATHS:
        u = base.rstrip("/") + path
        status, headers, text, err = http_probe(u, method="GET")
        if is_railway_application_not_found(status, headers, text):
            edge_paths.append(path)
            results[path] = "404 Railway Application not found"
        elif is_invalid_host_header(status, text):
            invalid_host_paths.append(path)
            results[path] = f"{status} Invalid Host header"
        else:
            results[path] = status if status is not None else f"error:{err}"
    mcp_status = results.get("/mcp")
    summary_bits = ", ".join(f"{p} -> {s}" for p, s in results.items())

    if edge_paths:
        return Finding(
            "Path / Address Tail Checker",
            "BLOCKER",
            "Railway's generic 'Application not found' page appeared during the path scan. That is a domain/service wiring problem first, not a path problem.",
            summary_bits,
            "Fix the Railway public domain on the deployed service first. After that, rerun this script to check whether the correct MCP path is '/mcp', '/sse', or something else.",
        )

    if invalid_host_paths:
        return Finding(
            "Path / Address Tail Checker",
            "BLOCKER",
            "At least one candidate endpoint reached the app but was rejected with 'Invalid Host header'. The path may be correct, but the app refuses this public hostname.",
            summary_bits,
            "Update the server's allowed/trusted hosts to include the exact Railway domain, redeploy, and rerun the checker. If '/mcp' is one of the affected paths, keep '/mcp' as the likely VS Code path after fixing host validation.",
        )

    if mcp_status == 404:
        # /mcp 404s but maybe another path works
        working = [
            p for p, s in results.items()
            if p != "/mcp" and status_implies_path_exists(s)
        ]
        if working:
            return Finding(
                "Path / Address Tail Checker",
                "BLOCKER",
                f"The '/mcp' part of the address returns 'not found', but these paths look real: {working}. Your MCP server is probably listening on a different path than '/mcp'.",
                summary_bits,
                f"In VS Code's MCP settings, change the URL's ending. Try ending it with one of: {working}. Many servers use '/sse' or just '/'.",
            )
        return Finding(
            "Path / Address Tail Checker",
            "BLOCKER",
            "The '/mcp' path returns 'not found' and no common alternative path answered either. Combined with the 'Application not found' message, the address is reaching Railway's edge but not your app.",
            summary_bits,
            "First fix the domain (Agent 3). The path likely can't answer until the domain is correctly attached.",
        )
    if isinstance(mcp_status, int) and mcp_status < 400:
        return Finding(
            "Path / Address Tail Checker",
            "OK",
            "The '/mcp' path answers normally.",
            summary_bits,
            "Path is fine.",
        )
    if mcp_status in (405, 406, 400, 401, 403, 415, 421, 422):
        return Finding(
            "Path / Address Tail Checker",
            "INFO",
            "The '/mcp' path exists but rejected a plain visit. That's actually normal — MCP endpoints often only accept special requests, not ordinary browser visits.",
            summary_bits,
            "This path is probably correct. The problem is more likely the request type/headers (see Transport agent).",
        )
    return Finding(
        "Path / Address Tail Checker",
        "WARNING",
        f"The '/mcp' path gave an unexpected response ({mcp_status}).",
        summary_bits,
        "Review alongside the other agents' findings.",
    )


def agent_transport(base):
    """AGENT 5: Streamable-HTTP vs SSE mismatch."""
    u = base.rstrip("/") + "/mcp"
    # MCP Streamable HTTP expects a POST with JSON and an Accept header that
    # allows both json and event-stream.
    init_body = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05",
                   "capabilities": {}, "clientInfo": {"name": "diag", "version": "1"}},
    })
    status, headers, text, err = http_probe(
        u, method="POST",
        extra_headers={"Content-Type": "application/json",
                       "Accept": "application/json, text/event-stream"},
        body=init_body,
    )
    if err:
        return Finding(
            "Transport Type Checker",
            "WARNING",
            "Could not test the proper MCP request because the address wasn't reachable.",
            f"Error: {err}",
            "Fix the domain/path first.",
        )
    if is_railway_application_not_found(status, headers, text):
        return Finding(
            "Transport Type Checker",
            "BLOCKER",
            "Even a correct MCP request returns 'not found'. This confirms the request never reaches your app — it's an address problem, not a transport problem.",
            f"POST /mcp -> {status}; body: {text[:200]}",
            "Your Railway logs show the server uses 'StreamableHTTP'. So the server speaks the right language — the address just isn't wired to it. Fix the domain in Railway.",
        )
    if is_invalid_host_header(status, text):
        return Finding(
            "Transport Type Checker",
            "BLOCKER",
            "A proper Streamable-HTTP MCP request reached the service, but the service rejected the Railway hostname with 'Invalid Host header'.",
            f"POST /mcp -> {status}; body: {text[:200]}",
            "Keep VS Code on the http/StreamableHTTP transport, but fix the server's allowed/trusted host configuration to include this Railway domain.",
        )
    if status == 404:
        return Finding(
            "Transport Type Checker",
            "BLOCKER",
            "A proper Streamable-HTTP MCP request reached a server, but '/mcp' was not found. That points to a wrong path, not a broken port.",
            f"POST /mcp -> {status}; body: {text[:200]}",
            "Use the Path / Address Tail Checker result to choose the correct URL ending in VS Code.",
        )
    if status and status < 400:
        ct = header_value(headers, "Content-Type")
        return Finding(
            "Transport Type Checker",
            "OK",
            f"A proper MCP request was accepted (response type: {ct}). The server speaks Streamable-HTTP correctly.",
            f"POST /mcp -> {status}, Content-Type {ct}",
            "If VS Code still fails, make sure VS Code is set to the 'http' (streamable) MCP type, not the old 'sse' type.",
        )
    return Finding(
        "Transport Type Checker",
        "INFO",
        f"The MCP request got status {status}. Inspect the evidence to interpret.",
        f"POST /mcp -> {status}; body: {text[:200]}",
        "Compare with what VS Code expects.",
    )


def agent_port_reality_check(parsed):
    """AGENT 6: The 8000-vs-8080 question, explained."""
    if parsed.port and parsed.port not in (443,):
        severity = "BLOCKER"
        evidence = f"The tested URL explicitly includes public port :{parsed.port}."
        suggestion = f"Remove ':{parsed.port}' from the VS Code URL. Use {parsed.scheme}://{parsed.hostname}{parsed.path or ''}."
    else:
        severity = "INFO"
        evidence = "The tested URL has no explicit public app port."
        suggestion = "In your app's start command, bind to the PORT variable (e.g. the server should read the PORT environment variable) instead of a fixed number. Then stop adding ':8000' or ':8080' to the URL in VS Code — use the plain https address."
    return Finding(
        "Port Myth-Buster",
        severity,
        "Switching between port 8000 and 8080 will almost never fix this. On Railway, the public web address always uses the normal secure port (443) on the outside. Inside, Railway tells your app which port to use through a setting called PORT. Your job is to make the app listen on whatever PORT Railway provides — not to pick 8000 or 8080 by hand. The public URL never includes a port number.",
        evidence,
        suggestion,
    )


# ---------------------------------------------------------------------------
# ADVERSARIAL VERIFIER — tries to DISPROVE each finding before we trust it.
# ---------------------------------------------------------------------------

def adversarial_verifier(findings, base):
    """
    For each finding, this agent actively tries to prove it WRONG by gathering
    independent evidence. Only findings that survive scrutiny are marked verified.
    """
    # Gather one fresh, independent snapshot to cross-check against.
    root_status, root_headers, root_text, root_err = http_probe(base, method="GET")
    root_says_not_found = is_railway_application_not_found(
        root_status, root_headers, root_text
    )
    root_rejects_host = is_invalid_host_header(root_status, root_text)

    for f in findings:
        if f.agent == "URL Shape Checker":
            f.verified = True
            f.verifier_note = "URL shape is determined locally from the text you supplied; accepted."

        elif f.agent == "Is The App Reachable":
            # Challenge: maybe it was a transient blip. Re-test independently.
            if f.severity == "BLOCKER":
                if root_err and "Could not reach" in f.plain_summary:
                    f.verified = True
                    f.verifier_note = f"Independently re-fetched the root address and still got a network error: {root_err}"
                elif root_rejects_host and "Host header" in f.plain_summary:
                    f.verified = True
                    f.verifier_note = "Independently re-fetched the root address and again saw a host-header rejection. Conclusion holds."
                elif root_says_not_found:
                    f.verified = True
                    f.verifier_note = "Independently re-fetched the root address; Railway's 'Application not found' message appeared again. Conclusion holds."
                else:
                    f.verified = False
                    f.verifier_note = "On recheck the root address did NOT show 'Application not found'. The earlier failure may be intermittent or path-specific — do not over-trust this BLOCKER."
            else:
                # Claimed OK. Challenge: make sure it isn't secretly the not-found page with a 200.
                if root_says_not_found:
                    f.verified = False
                    f.verifier_note = "Contradiction: a recheck of the root DID show Railway's not-found page. Demoting confidence."
                else:
                    f.verified = True
                    f.verifier_note = "Recheck agrees: no Railway not-found page at the root."

        elif f.agent == "Path / Address Tail Checker":
            # Challenge: if everything 404s, is it really a path issue or a domain issue?
            if f.severity == "BLOCKER" and root_says_not_found:
                f.verified = True
                f.verifier_note = "Cross-checked against root domain, which also shows Railway not-found. The path finding is consistent with an underlying DOMAIN problem — fix domain first; path advice is secondary."
            elif f.severity == "BLOCKER" and root_err:
                f.verified = True
                f.verifier_note = "Root recheck also failed with a network error, so path conclusions may be blocked by reachability. Fix reachability first, then rerun."
            else:
                f.verified = True
                f.verifier_note = "Path scan results are reported as raw status codes, which are directly observable facts. Accepted."

        elif f.agent == "Transport Type Checker":
            # Challenge: don't claim transport works if the domain is dead.
            if f.severity == "OK" and (root_says_not_found or root_err):
                f.verified = False
                f.verifier_note = "Suspicious: transport reported OK but the root recheck shows a reachability/domain problem. Re-examine; this OK may be a fluke."
            else:
                f.verified = True
                f.verifier_note = "Transport conclusion is consistent with observed status codes."

        elif f.agent == "DNS Resolver":
            # DNS is a hard fact; just confirm it didn't change.
            try:
                socket.gethostbyname(urlparse(base).hostname)
                resolves = True
            except Exception:
                resolves = False
            if (f.severity == "OK") == resolves:
                f.verified = True
                f.verifier_note = "DNS re-checked and matches the original finding."
            else:
                f.verified = False
                f.verifier_note = "DNS result changed between checks — treat as unstable."

        elif f.agent == "Secure Connection (TLS)":
            f.verified = True
            f.verifier_note = "TLS result is a direct observation of the handshake; accepted as-is."

        elif f.agent == "Port Myth-Buster":
            # This is an explanatory/educational finding. Verify the factual claim:
            # the public URL has no explicit port and still reaches an edge.
            parsed = urlparse(base)
            no_explicit_port = parsed.port is None
            if f.severity == "BLOCKER" and not no_explicit_port:
                f.verified = True
                f.verifier_note = "Confirmed: the tested public URL includes an explicit port, which should be removed for Railway's public domain."
            elif no_explicit_port:
                f.verified = True
                f.verifier_note = "Confirmed: the working public address carries no port number, supporting the explanation that hand-picking 8000/8080 is not the fix."
            else:
                f.verified = False
                f.verifier_note = "The URL being tested includes an explicit port; revisit the explanation in light of that."

        else:
            f.verified = True
            f.verifier_note = "No specific challenge defined; accepted."

    return findings


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run(url, json_out="last_report.json"):
    url, parsed = normalize_url(url)
    base = base_url(parsed)

    print("=" * 70)
    print("RAILWAY MCP CONNECTION DIAGNOSTIC")
    print(f"Target: {url}")
    print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)
    print("\nDispatching probe agents...\n")

    findings = []
    findings.append(agent_url_shape(parsed))
    findings.append(agent_dns(parsed))
    findings.append(agent_tls(parsed))
    findings.append(agent_app_alive(base))
    findings.append(agent_path_scan(base))
    findings.append(agent_transport(base))
    findings.append(agent_port_reality_check(parsed))

    print("Probe agents finished. Running ADVERSARIAL VERIFIER...\n")
    findings = adversarial_verifier(findings, base)

    # Report
    order = {"BLOCKER": 0, "WARNING": 1, "INFO": 2, "OK": 3}
    findings.sort(key=lambda f: order.get(f.severity, 9))

    for f in findings:
        mark = {True: "VERIFIED", False: "DISPUTED BY VERIFIER", None: "unverified"}[f.verified]
        print("-" * 70)
        print(f"[{f.severity}] {f.agent}   ({mark})")
        print(f"  What it means : {f.plain_summary}")
        print(f"  Evidence      : {f.evidence}")
        print(f"  What to do    : {f.suggestion}")
        if f.verifier_note:
            print(f"  Verifier says : {f.verifier_note}")
    print("-" * 70)

    # Plain-English bottom line
    blockers = [f for f in findings if f.severity == "BLOCKER" and f.verified]
    print("\n" + "=" * 70)
    print("PLAIN-ENGLISH BOTTOM LINE")
    print("=" * 70)
    if blockers:
        print("The most likely problem, confirmed by the verifier, is:\n")
        for b in blockers:
            print(f"  * {b.plain_summary}")
            print(f"    -> {b.suggestion}\n")
    else:
        disputed = [f for f in findings if f.verified is False]
        print("No fully-verified blocker was found in this run.")
        if disputed:
            print("Some findings were disputed by the verifier and need a second look:")
            for d in disputed:
                print(f"  * {d.agent}: {d.verifier_note}")

    # Save machine-readable copy
    out = {"target": url, "findings": [f.to_dict() for f in findings]}
    report_path = Path(json_out).expanduser()
    try:
        if report_path.parent != Path("."):
            report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n(Full machine-readable report saved to {report_path})")
    except OSError as e:
        print(f"\n(Could not save the machine-readable report: {e})")

    return findings


def self_test():
    """Fast mocked tests for the diagnostic decision tree."""
    global http_probe

    original_http_probe = http_probe

    def install_fake_probe(responses):
        def fake_probe(url, method="GET", extra_headers=None, timeout=10, body=None):
            parsed = urlparse(url)
            path = parsed.path or "/"
            key = (method, path)
            fallback_key = (method, "*")
            value = responses.get(key, responses.get(fallback_key))
            if value is None:
                return 404, {"Server": "fake-app"}, "not found", None
            return value

        globals()["http_probe"] = fake_probe

    def assert_finding(finding, severity, contains):
        assert finding.severity == severity, finding.to_dict()
        assert contains in finding.plain_summary, finding.to_dict()

    try:
        normalized, parsed = normalize_url("example.up.railway.app")
        assert normalized == "https://example.up.railway.app"
        assert parsed.hostname == "example.up.railway.app"

        _, port_parsed = normalize_url("https://example.up.railway.app:8080/mcp")
        port_finding = agent_url_shape(port_parsed)
        assert_finding(port_finding, "BLOCKER", "explicit port")

        install_fake_probe({
            ("GET", "*"): (
                404,
                {"Server": "railway-edge"},
                "Application not found",
                None,
            ),
            ("POST", "/mcp"): (
                404,
                {"Server": "railway-edge"},
                "Application not found",
                None,
            ),
        })
        edge = agent_app_alive("https://fake.up.railway.app")
        assert_finding(edge, "BLOCKER", "Railway itself says")
        verified = adversarial_verifier([edge], "https://fake.up.railway.app")[0]
        assert verified.verified is True, verified.to_dict()

        install_fake_probe({
            ("GET", "/"): (404, {"Server": "fake-app"}, "app route missing", None),
            ("GET", "/mcp"): (405, {"Server": "fake-app"}, "method not allowed", None),
            ("GET", "/sse"): (404, {"Server": "fake-app"}, "not found", None),
            ("GET", "/messages"): (404, {"Server": "fake-app"}, "not found", None),
            ("GET", "/health"): (200, {"Server": "fake-app"}, "ok", None),
            ("POST", "/mcp"): (
                200,
                {"Content-Type": "application/json"},
                '{"jsonrpc":"2.0","id":1,"result":{}}',
                None,
            ),
        })
        app_404 = agent_app_alive("https://fake.up.railway.app")
        assert_finding(app_404, "OK", "NOT Railway")
        path = agent_path_scan("https://fake.up.railway.app")
        assert_finding(path, "INFO", "'/mcp' path exists")
        transport = agent_transport("https://fake.up.railway.app")
        assert_finding(transport, "OK", "proper MCP request")

        install_fake_probe({
            ("GET", "/"): (404, {"Server": "railway-hikari"}, "Not Found", None),
            ("GET", "/mcp"): (
                421,
                {"Server": "railway-hikari"},
                "Invalid Host header",
                None,
            ),
            ("GET", "/sse"): (404, {"Server": "railway-hikari"}, "Not Found", None),
            ("GET", "/messages"): (404, {"Server": "railway-hikari"}, "Not Found", None),
            ("GET", "/health"): (404, {"Server": "railway-hikari"}, "Not Found", None),
            ("POST", "/mcp"): (
                421,
                {"Server": "railway-hikari"},
                "Invalid Host header",
                None,
            ),
        })
        host_path = agent_path_scan("https://fake.up.railway.app")
        assert_finding(host_path, "BLOCKER", "Invalid Host header")
        host_transport = agent_transport("https://fake.up.railway.app")
        assert_finding(host_transport, "BLOCKER", "rejected the Railway hostname")

        install_fake_probe({
            ("GET", "*"): (None, {}, "", "timed out"),
            ("POST", "/mcp"): (None, {}, "", "timed out"),
        })
        network = agent_app_alive("https://fake.up.railway.app")
        assert_finding(network, "BLOCKER", "Could not reach")
        verified = adversarial_verifier([network], "https://fake.up.railway.app")[0]
        assert verified.verified is True, verified.to_dict()

    finally:
        globals()["http_probe"] = original_http_probe

    print("Self-test passed: diagnostic workflow classifications are behaving.")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Diagnose Railway-hosted MCP connection issues from VS Code."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_URL,
        help="Railway URL to test. Bare domains are treated as https:// domains.",
    )
    parser.add_argument(
        "--json-out",
        default="last_report.json",
        help="Where to save the machine-readable JSON report.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mocked local workflow tests instead of making network requests.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.self_test:
        self_test()
    else:
        try:
            run(args.url, json_out=args.json_out)
        except ValueError as e:
            print(f"Invalid URL: {e}", file=sys.stderr)
            sys.exit(2)
