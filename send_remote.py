#!/usr/bin/env python3
"""Stream a depthbrush .gcode layer to the GRBL plotter server (remote mode).

Follows the server's JSON-over-WebSocket protocol: enables remote mode, sends
one intent-level line per external_command, and paces itself against
external_progress / status.external_queue_size so the queue stays shallow.

  python3 send_remote.py out/garden/02_near_pen.gcode --host localhost --port 8000
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

WINDOW = 24  # max commands in flight (sent - executed)


async def stream(path: str, host: str, port: int, dry_run: bool = False):
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(";")]
    print(f"{len(lines)} commands from {path}")
    if dry_run:
        for ln in lines[:20]:
            print(" ", ln)
        print("  ... (dry run)")
        return

    uri = f"ws://{host}:{port}/ws"
    async with websockets.connect(uri, max_size=2 ** 22) as ws:
        executed = 0
        sent = 0
        done = asyncio.Event()

        async def reader():
            nonlocal executed
            async for raw in ws:
                msg = json.loads(raw)
                t = msg.get("type")
                if t == "hello":
                    caps = msg.get("capabilities", {})
                    print(f"connected: {caps.get('server_name', '?')} "
                          f"protocol {caps.get('protocol_version', '?')}")
                elif t == "external_progress":
                    executed = msg.get("commands_executed", executed)
                    if executed % 50 == 0 or executed == len(lines):
                        print(f"  executed {executed}/{len(lines)}", end="\r")
                    if executed >= len(lines):
                        done.set()
                elif t == "external_command_rejected":
                    print(f"\nREJECTED: {msg.get('command')} — {msg.get('reason')}")
                    done.set()
                elif t == "external_error":
                    print(f"\nERROR on {msg.get('command')}: {msg.get('error')}")

        reader_task = asyncio.create_task(reader())
        await ws.send(json.dumps({"type": "toggle_remote_mode", "enabled": True}))
        await asyncio.sleep(0.5)  # let the server reset its pipeline

        for ln in lines:
            while sent - executed >= WINDOW:
                await asyncio.sleep(0.05)
            await ws.send(json.dumps({"type": "external_command", "command": ln}))
            sent += 1

        print(f"\nall {sent} commands queued; waiting for execution...")
        await done.wait()
        await ws.send(json.dumps({"type": "get_status"}))
        await asyncio.sleep(0.5)
        reader_task.cancel()
        print("layer complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gcode")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(stream(args.gcode, args.host, args.port, args.dry_run))
    except KeyboardInterrupt:
        print("\ninterrupted — brush state is whatever the last executed command left it!")
        sys.exit(1)


if __name__ == "__main__":
    main()
