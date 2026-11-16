# Free Model Aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/nisarena2/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/nisarena2/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that puts the best free models behind a single API. It speaks
both wires — OpenAI Chat Completions (`POST /v1/chat/completions`) and
Anthropic Messages (`POST /v1/messages`), JSON and streaming — and every
completion it answers comes from one of the fourteen free OpenRouter models,
dialled through the grader's gateway on the trial's key.

## How it picks a model

Each prompt is served by the highest-rated free model that meets what the
prompt needs:

- JSON mode → the best model that supports JSON (Nemotron 3.5 Super VL,
  rating 1303, then Gemma 4 31B at 1210).
- Tools → the best model that supports tools (Inkling 2 Small, 1475).
- Anything else → the best free model overall (Inkling 2 Small, 1475).

A refusal (429 or 5xx) is retried once on the same model, then failed over
to the next eligible model, at most six gateway calls per prompt; the
refusal trail is reported in the `x-makefun-upstream-failures` header. A
prompt that is given up on scores nothing, so the server keeps failing over
until the eligible list is spent.

## Wires

- `GET /v1/models` answers the OpenAI list for `Authorization: Bearer`
  and the Anthropic list for `x-api-key`; both list exactly the fourteen
  free ids, once each.
- Requests name `model: "auto"`. The server replaces it with the dialable
  id it chose and puts that id back on every answer and streamed chunk.
- Anthropic requests are translated to Chat Completions for the gateway
  call (system prompt, tools, tool choices, stop sequences) and the
  gateway's answer is translated back, usage and stop reason whole.
- Streaming is synthesised locally: the upstream call is always
  non-streaming, and the server emits the events in the wire's order,
  with a whole usage count at the end.

## Running

```sh
python3 main.py --list   # print the catalog and exit
python3 main.py          # listen, register with $MAKEFUN_ENV_URL, serve
```

The server picks a free port, registers it with the env, polls
`GET $MAKEFUN_ENV_URL/work`, and exits 0 when the env says done. It
never holds a provider key: the trial's key arrives in `OPENAI_API_KEY`
and is sent as the vendor's SDK sends it; the gateway swaps it for the
entrant's own OpenRouter key.

## Files

- `main.py` — the whole server, Python 3.13 standard library only.
- `README.md` — this file.
- `LICENSE` — MIT.
