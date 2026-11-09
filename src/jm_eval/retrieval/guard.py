"""Network guard for scripts that read SUSEP data (ground rule 1: it never leaves this machine).

`enforce_local_only()` is called before any private file is opened. It
1. refuses to run if a cloud client is loaded in the process or a proxy is configured;
2. puts the Hugging Face libraries in offline mode;
3. makes every network connection of the process fail: internet sockets cannot connect and host
   names other than this machine's cannot be looked up.
A later attempt to reach the network therefore raises instead of sending anything.
"""

from __future__ import annotations

import os
import socket
import sys

# Modules that talk to cloud services. A script that reads SUSEP data must not have them loaded.
CLOUD_MODULES = ("google.genai", "google.cloud", "google.auth", "google.oauth2", "googleapiclient", "vertexai",
                 "openai", "anthropic", "boto3", "botocore", "azure", "cohere", "mistralai",
                 "jm_eval.llm", "jm_eval.retrieval.vertex_embed", "jm_eval.server")
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY")
LOCAL_HOSTS = (None, "", "localhost", "127.0.0.1", "::1")


class NetworkBlocked(RuntimeError):
    """Raised when guarded code tries to open a network connection."""


def loaded_cloud_modules() -> list[str]:
    return sorted(name for name in sys.modules
                  if any(name == m or name.startswith(m + ".") for m in CLOUD_MODULES))


def proxy_settings() -> list[str]:
    return sorted(v for name in PROXY_VARIABLES for v in (name, name.lower()) if os.environ.get(v))


def assert_local_only() -> None:
    problems = []
    if (modules := loaded_cloud_modules()):
        problems.append(f"cloud client modules are loaded: {modules[:5]}")
    if (proxies := proxy_settings()):
        problems.append(f"proxy environment variables are set: {proxies}")
    if problems:
        raise RuntimeError("refusing to read SUSEP data: " + "; ".join(problems))


def block_network() -> None:
    """Make internet sockets of this process unable to connect and remote names unresolvable.

    Local Unix sockets and look-ups of this machine's own name still work.
    """
    if getattr(socket.socket, "_jm_eval_blocked", False):
        return
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if name not in LOCAL_HOSTS and name != socket.gethostname():
            raise NetworkBlocked(f"network access is blocked while SUSEP data is processed (look-up of {name!r})")
        return original_getaddrinfo(host, *args, **kwargs)

    def _refuse(sock: socket.socket, address) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"network access is blocked while SUSEP data is processed (tried {address!r})")

    def connect(self, address):
        _refuse(self, address)
        return original_connect(self, address)

    def connect_ex(self, address):
        _refuse(self, address)
        return original_connect_ex(self, address)

    socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
    socket.getaddrinfo = getaddrinfo
    socket.socket._jm_eval_blocked = True
    socket.socket._jm_eval_original = (original_connect, original_connect_ex, original_getaddrinfo)


def unblock_network() -> None:
    """Undo `block_network` (tests only)."""
    if getattr(socket.socket, "_jm_eval_blocked", False):
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = socket.socket._jm_eval_original
        socket.socket._jm_eval_blocked = False


def enforce_local_only() -> None:
    assert_local_only()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    block_network()
