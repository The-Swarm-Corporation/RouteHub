# Test RouteHub response handling offline

Test your response-handling code before you connect it to a model provider. This guide turns [`mock_response`](../README.md#testing-without-a-provider) into a runnable pytest suite. You will check ordinary responses, empty replies, synchronous and asynchronous streams, usage-only chunks, and exception handling. A network guard makes an unexpected socket connection fail the test instead of silently reaching a provider.

These are application contract tests. They check how your code consumes RouteHub results. They do not measure model quality, verify credentials, or establish that a provider accepts your request parameters.

## Prepare an isolated environment

Use Python 3.10 or later. From the RouteHub repository root, install the package and development dependencies:

```bash
uv sync
```

Alternatively, create a virtual environment and install the checkout with the test dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e . "pytest>=9.1.1" "pytest-asyncio>=1.4.0"
```

Package installation needs internet access. The tests below do not need a provider account or API key. They import this checkout, so they exercise the code you are developing rather than a separately pinned package. The examples target RouteHub 0.4.0; rerun them when you upgrade.

## Reject unexpected network access

Read the module-scoped `offline_only` fixture in [`tests/test_offline_response_handling.py`](../tests/test_offline_response_handling.py). It runs after the shared `fake_env` fixture, clears the fake API keys too, and rejects socket operations for this module only.

The fixture runs automatically for every test in this example module. The repository's shared fixtures still isolate caches and supply a fake model catalog. Pytest restores the patched functions and environment variables after each test. It does not edit your shell profile, `.env` files, or saved credentials.

Keep provider calls out of module-level imports and setup code that runs before this fixture. This guard covers Python socket operations during each test; it is not an operating-system network policy and does not cover subprocesses or native transports. Use your CI runner's network controls if you need process-wide isolation.

## Run the contract tests

The complete runnable cases are in [`tests/test_offline_response_handling.py`](../tests/test_offline_response_handling.py).

Run the example from the RouteHub repository root:

```bash
uv run pytest -q tests/test_offline_response_handling.py
# With the pip environment instead: python -m pytest -q tests/test_offline_response_handling.py
```

The expected result is **12 passed**. Parameterized cases count separately, including empty synchronous and asynchronous streams. `asyncio.run()` drives each asynchronous example, so each asynchronous example has its own event loop. The repository's full suite also uses `pytest-asyncio`, which is included in its development dependencies. The final test checks the guard itself: its patched function raises before a connection is attempted.

## Choose assertions that survive upgrades

Assert the content and fields that your application uses. The mock generates response identifiers and creation timestamps at runtime, so a snapshot of the entire object would change even when your application behavior stays correct. Likewise, do not use mock token counts to estimate a production bill: they are approximations.

Streaming responses need more care than a single response. Some chunks carry a role or finish reason without text. A final usage chunk can have an empty `choices` list. The `chunk_text()` helper checks that list before indexing it, and the stream test includes a usage chunk to exercise that branch. Compare the reconstructed text, including whitespace, rather than requiring a particular number of chunks or provider-specific token boundaries.

Test asynchronous response handling separately from synchronous handling. Await `acompletion()` first, then iterate its streaming result with `async for`. The context managers in the examples also exercise the stream cleanup interface your application should use when iteration ends early.

## Keep mock and provider tests separate

Passing an exception as `mock_response` raises that exception directly. The rate-limit test verifies that application code can catch the expected error type and inspect its status. It does not exercise RouteHub's HTTP retry loop, provider error translation, fallback configuration, or a real timeout. Testing those behaviors needs a controlled fake HTTP transport or a separately configured integration test.

Similarly, returning a JSON string does not enforce a provider schema. The JSON cases verify your own parsing behavior, including malformed output. Add your application's schema validator after parsing when your contract requires specific fields or types. See [Structured output](../README.md#structured-output) for provider-facing configuration.

Mock calls bypass model resolution. A successful mock for `MODEL` does not establish that the model exists or that a provider supports it. Check routing separately using the [model catalog](../README.md#model-catalog), and keep any real-provider smoke test outside this offline suite with explicit credentials and a deliberate invocation. Do not automatically rerun a failing mock test against a paid provider.

When you adapt this suite, replace the small parsing helpers with calls to your application's actual response-handling functions. Keep the network guard, empty-response cases, usage-only chunk handling, and malformed-output checks so that regressions fail locally before you opt into an integration run.

## Check both supported SDK versions

The module is collected by the normal repository suite. Run it together with the existing tests against both supported OpenAI SDK lines:

```bash
uv run pytest
uv run --with "openai<3" pytest
uv run ruff check .
```

These commands do not invoke `tests/live_smoke.py`. That script is a separate opt-in provider integration check and requires credentials.
