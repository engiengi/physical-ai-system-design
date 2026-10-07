"""Probe a chosen policy endpoint without invoking inference."""
from .settings import URI
import argparse
import json
import socket
import subprocess
from urllib.parse import urlsplit


def probe(uri, timeout=3.0, handshake=False):
    parsed = urlsplit(uri)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
        raise ValueError("Use a ws:// or wss:// policy URI")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use COSMOS3_API_TOKEN for authentication; do not put credentials in the URI")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    report = {"uri": uri, "host": parsed.hostname, "port": port}
    result = subprocess.run(["ping", "-c", "4", "-W", "2", parsed.hostname],
                            capture_output=True, text=True, timeout=15)
    report["ping_ok"] = result.returncode == 0
    report["ping_output"] = result.stdout + result.stderr
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout) as sock:
            report.update(tcp_ok=True, local_ip=sock.getsockname()[0])
    except OSError as exc:
        report.update(tcp_ok=False, tcp_error=str(exc))
    if handshake and report["tcp_ok"]:
        import os
        from robolab.eval.websocket_transport import MsgPackWebSocketTransport
        transport = MsgPackWebSocketTransport(uri, api_token=os.environ.get("COSMOS3_API_TOKEN"),
            metadata_timeout=timeout, connect_kwargs={"open_timeout": timeout, "close_timeout": 2})
        try:
            transport.connect()
            report["openpi_handshake_ok"] = True
        except Exception as exc:
            report.update(openpi_handshake_ok=False, handshake_error=str(exc))
        finally:
            transport.close()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=URI)
    parser.add_argument("--handshake", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = probe(args.uri, handshake=args.handshake)
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n")
    raise SystemExit(0 if report.get("openpi_handshake_ok", report["tcp_ok"]) else 1)


if __name__ == "__main__":
    main()
