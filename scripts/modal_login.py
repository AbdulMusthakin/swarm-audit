#!/usr/bin/env python3
"""Modal browser sign-in that survives dropped connections while waiting.

Same steps as `modal token new`, but the wait for approval reconnects on errors
instead of exiting. Prints the link, never the token; stores it in ~/.modal.toml.
"""
import asyncio
import time

from modal.client import _Client
from modal.config import config
from modal.token_flow import _set_token, _TokenFlow


async def wait_once(server_url: str, flow_id: str, secret: str):
    async with _Client.anonymous(server_url) as c:
        tf = _TokenFlow(c)
        tf.token_flow_id, tf.wait_secret = flow_id, secret
        return await tf.finish()


async def main() -> None:
    server_url = config.get("server_url")
    async with _Client.anonymous(server_url) as client:
        tf = _TokenFlow(client)
        async with tf.start("swarm-audit", None, None) as (_, web_url, code):
            print(f"LINK {web_url}", flush=True)
            if code:
                print(f"CODE {code}", flush=True)
            flow_id, secret = tf.token_flow_id, tf.wait_secret
            deadline, result = time.time() + 1800, None
            while result is None and time.time() < deadline:
                try:
                    result = await wait_once(server_url, flow_id, secret)
                except Exception as e:  # dropped connection: reconnect and keep waiting
                    print(f"wait interrupted ({type(e).__name__}); reconnecting", flush=True)
                    await asyncio.sleep(2)
    if result is None:
        raise SystemExit("sign-in not approved within 30 minutes")
    print(f"APPROVED workspace={result.workspace_username}", flush=True)
    await _set_token(result.token_id, result.token_secret, server_url=server_url)
    print("TOKEN STORED", flush=True)


import modal

with modal.enable_output():
    asyncio.run(main())
