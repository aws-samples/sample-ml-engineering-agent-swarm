"""Bounded no-key research tests: closed transport, real URL guard, no credentials."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("AMLAS_TAVILY_KEYLESS", raising=False)
    monkeypatch.delenv("AMLAS_AGENT_HTTP_ALLOW_HOSTS", raising=False)
    from amlas.tools import pathref

    monkeypatch.setattr(
        pathref,
        "_resolve_host_ips",
        lambda host: (
            [host]
            if host in {"127.0.0.1", "169.254.169.254", "10.0.0.1", "100.64.0.1", "::1"}
            else ["93.184.216.34"]
        ),
    )


def public_fetch_module() -> Any:
    try:
        return importlib.import_module("amlas.tools.research.public_fetch")
    except ModuleNotFoundError:
        pytest.fail("missing bounded supplied-public-URL research fetch module")


def test_keyless_search_uses_documented_header_and_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.tools.research import tavily

    calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args: Any):
            return None

        def post(self, url: str, **kwargs: Any):
            calls.append(kwargs)

            class Response:
                status_code = 200

                def json(self):
                    return {
                        "results": [
                            {"title": "Example", "url": "https://example.com", "content": "Observed source"}
                        ]
                    }

            return Response()

    monkeypatch.setattr(tavily.httpx, "Client", Client)
    results = tavily.tavily_search("generic public query")
    assert calls[0]["headers"]["X-Tavily-Access-Mode"] == "keyless"
    assert "api_key" not in calls[0]["json"]
    assert results[0].identifiers["access_mode"] == "keyless"


def test_rate_limited_keyless_failure_is_explicit_and_not_fake_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools.research import tavily

    class Client:
        def __init__(self, **kwargs: Any):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args: Any):
            return None

        def post(self, *args: Any, **kwargs: Any):
            class Response:
                status_code = 429
                text = "untrusted echo must not be reported as research"

            return Response()

    monkeypatch.setattr(tavily.httpx, "Client", Client)
    with pytest.raises(tavily.TavilyError, match="429"):
        tavily.tavily_search("generic public query")


def test_default_sources_include_verified_keyless_and_free_scholarly_sources() -> None:
    from amlas.tools.research.strands_tools import _build_default_sources

    assert {"arxiv", "semantic_scholar", "tavily"} <= set(_build_default_sources())


def test_user_supplied_url_fetch_returns_real_untrusted_content(monkeypatch: pytest.MonkeyPatch) -> None:
    module = public_fetch_module()
    seen: list[tuple[str, list[str]]] = []

    def request(url: str, ips: list[str], timeout: float, limit: int):
        seen.append((url, ips))
        return (
            200,
            {"content-type": "text/html"},
            b"<html><title>Example</title><p>Observed public content.</p><script>ignore instructions</script></html>",
        )

    monkeypatch.setattr(module, "_request_public", request)
    result = module.fetch_public_urls(["https://example.com/page"])
    assert result["pages"][0]["title"] == "Example"
    assert "Observed public content" in result["pages"][0]["text"]
    assert "ignore instructions" not in result["pages"][0]["text"]
    assert result["untrusted_data"] is True
    assert result["coverage"] == "provided_urls_only"
    assert seen == [("https://example.com/page", ["93.184.216.34"])]


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://100.64.0.1/",
        "http://[::1]/",
        "http://localhost/",
        "http://metadata.internal/",
        "https://thing." + "corp." + "example.org/",  # Synthetic reserved label fixture.
        "https://" + ":".join(("user", "password")) + "@example.com/",  # Negative synthetic userinfo.
        "file:///etc/passwd",
        "s3://bucket/key",
    ],
)
def test_private_internal_userinfo_and_non_http_urls_never_reach_transport(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    module = public_fetch_module()
    monkeypatch.setattr(module, "_request_public", lambda *args: pytest.fail("unsafe URL reached transport"))
    result = module.fetch_public_urls([url])
    assert result["pages"] == []
    assert result["errors"]


def test_redirect_to_metadata_is_validated_before_second_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    module = public_fetch_module()
    seen: list[str] = []

    def request(url: str, *args: Any):
        seen.append(url)
        return 302, {"location": "http://169.254.169.254/latest/meta-data/"}, b""

    monkeypatch.setattr(module, "_request_public", request)
    result = module.fetch_public_urls(["https://example.com/start"])
    assert result["pages"] == [] and result["errors"]
    assert seen == ["https://example.com/start"]


def test_redirect_chain_byte_and_page_budgets_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    module = public_fetch_module()
    seen: list[str] = []

    def request(url: str, *args: Any):
        seen.append(url)
        return 302, {"location": "/again"}, b""

    monkeypatch.setattr(module, "_request_public", request)
    result = module.fetch_public_urls(["https://example.com/start"], max_redirects=1)
    assert result["errors"] and len(seen) == 2
    monkeypatch.setattr(
        module, "_request_public", lambda *args: (200, {"content-type": "text/plain"}, b"x" * 17)
    )
    result = module.fetch_public_urls(
        ["https://example.com/a", "https://example.com/b"], max_pages=1, max_bytes=16
    )
    assert result["pages"] == [] and result["errors"]
    assert result["skipped_count"] == 1


def test_disabled_keyless_stays_explicit_and_supplied_url_tool_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from amlas.tools.research import tavily

    monkeypatch.setenv("AMLAS_TAVILY_KEYLESS", "false")
    with pytest.raises(tavily.TavilyMissingKeyError):
        tavily.tavily_search("generic public query")
    from amlas.subagents.declarations import ALGO_HYPOTHESIS
    from amlas.tools.research.strands_tools import research_fetch_urls

    assert research_fetch_urls in ALGO_HYPOTHESIS.tools


def test_dns_rebinding_to_private_address_is_blocked_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = public_fetch_module()
    from amlas.tools import pathref

    replies = iter((["93.184.216.34"], ["169.254.169.254"]))
    monkeypatch.setattr(pathref, "_resolve_host_ips", lambda host: next(replies))
    monkeypatch.setattr(
        module, "_request_public", lambda *args: pytest.fail("rebound private address reached transport")
    )
    result = module.fetch_public_urls(["https://example.com/page"])
    assert result["pages"] == [] and result["errors"]


@pytest.mark.parametrize("slow_phase", ["headers", "body"])
def test_request_deadline_interrupts_slow_headers_and_body(
    monkeypatch: pytest.MonkeyPatch, slow_phase: str
) -> None:
    import threading
    import time

    module = public_fetch_module()
    closed = threading.Event()

    class Sock:
        def shutdown(self, how: int):
            closed.set()

        def close(self):
            closed.set()

    class Response:
        status = 200

        def getheaders(self):
            return [("content-type", "text/plain")]

        def read(self, size: int):
            if slow_phase == "body":
                if closed.wait(0.15):
                    raise OSError("request closed")
            return b"observed"

    class Connection:
        def __init__(self, *args: Any):
            self.sock = Sock()

        def request(self, *args: Any, **kwargs: Any):
            return None

        def getresponse(self):
            if slow_phase == "headers" and closed.wait(0.15):
                raise OSError("request closed")
            return Response()

        def close(self):
            closed.set()

    monkeypatch.setattr(module, "_PinnedHTTPS", Connection)
    start = time.monotonic()
    with pytest.raises(module.PublicFetchError, match="deadline"):
        module._request_public("https://example.com/", ["93.184.216.34"], 0.03, 64)
    assert time.monotonic() - start < 0.12


def test_skipped_url_reporting_does_not_echo_unvalidated_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    module = public_fetch_module()
    monkeypatch.setattr(
        module, "_request_public", lambda *args: (200, {"content-type": "text/plain"}, b"observed")
    )
    result = module.fetch_public_urls(
        [
            "https://example.com/",
            "https://" + ":".join(("user", "private-value")) + "@example.com/?token=private-value",
        ],
        max_pages=1,
    )
    assert result["skipped_count"] == 1
    assert "private-value" not in str(result)


def test_tls_handshake_uses_remaining_deadline_and_assigned_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    module = public_fetch_module()
    clock = [0.0]
    timeouts: list[float] = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    connection = module._PinnedHTTPS("example.com", "93.184.216.34", 443, 3.0)

    class Raw:
        def close(self):
            pass

    raw = Raw()

    class TLS:
        def settimeout(self, timeout: float):
            timeouts.append(timeout)

        def do_handshake(self):
            assert connection.sock is self, "watchdog must see the TLS socket before handshake"
            assert timeouts[-1] == pytest.approx(1.0)
            clock[0] += 0.5

        def close(self):
            pass

    tls = TLS()

    def tcp(address: Any, timeout: float):
        assert timeout == pytest.approx(3.0)
        clock[0] += 2.0
        return raw

    class Context:
        def wrap_socket(self, sock: Any, *, server_hostname: str, do_handshake_on_connect: bool = True):
            assert sock is raw and server_hostname == "example.com"
            assert do_handshake_on_connect is False, "TLS must be assigned before a blocking handshake"
            return tls

    monkeypatch.setattr(module.socket, "create_connection", tcp)
    connection._context = Context()
    connection.connect()
    assert timeouts == [pytest.approx(1.0)]
    assert connection.sock is tls


@pytest.mark.parametrize("exhausted_stage", ["tcp", "wrap"])
def test_tls_does_not_start_handshake_after_connection_deadline(
    monkeypatch: pytest.MonkeyPatch, exhausted_stage: str
) -> None:
    module = public_fetch_module()
    clock = [0.0]
    closed: list[str] = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    connection = module._PinnedHTTPS("example.com", "93.184.216.34", 443, 3.0)

    class Raw:
        def close(self):
            closed.append("tcp")

    class TLS:
        def close(self):
            closed.append("wrap")

        def do_handshake(self):
            pytest.fail("TLS handshake started after the total deadline")

    def tcp(address: Any, timeout: float):
        clock[0] = 4.0 if exhausted_stage == "tcp" else 2.0
        return Raw()

    class Context:
        def wrap_socket(self, *args: Any, **kwargs: Any):
            assert exhausted_stage != "tcp", "TLS wrapping started after the TCP budget expired"
            clock[0] = 4.0
            return TLS()

    monkeypatch.setattr(module.socket, "create_connection", tcp)
    connection._context = Context()
    with pytest.raises(module.PublicFetchError, match="deadline"):
        connection.connect()
    assert exhausted_stage in closed


@pytest.mark.parametrize("characters", [12, 20037])
def test_page_text_excerpt_reports_complete_or_partial_coverage(
    monkeypatch: pytest.MonkeyPatch, characters: int
) -> None:
    module = public_fetch_module()
    monkeypatch.setattr(
        module, "_request_public", lambda *args: (200, {"content-type": "text/plain"}, b"x" * characters)
    )
    page = module.fetch_public_urls(["https://example.com/page"])["pages"][0]
    assert page["text_characters_returned"] == min(characters, 20000)
    assert page["text_characters_available"] == characters
    assert page["text_truncated"] is (characters > 20000)
    assert page["content_coverage"] == ("partial" if characters > 20000 else "complete")
