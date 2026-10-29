import argparse
import asyncio
import json
import queue
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from gemma_runtime import GemmaScheduler

ROOT = Path(__file__).resolve().parents[1]


def make_app(
    binary,
    model,
    context_limit=4096,
    max_batch=8,
    max_queue=128,
    full_prefill=False,
    batch_kv=True,
    all_batch_kv=False,
    gather_ms=2.0,
):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from transformers import AutoTokenizer

    @asynccontextmanager
    async def lifespan(app):
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        scheduler = GemmaScheduler(
            binary,
            model,
            tokenizer,
            context_limit,
            max_batch,
            max_queue,
            full_prefill=full_prefill,
            batch_kv=batch_kv,
            all_batch_kv=all_batch_kv,
            gather_ms=gather_ms,
        )
        app.state.scheduler = scheduler
        try:
            yield
        finally:
            await asyncio.to_thread(scheduler.close)

    app = FastAPI(lifespan=lifespan)
    name = "mini-gemma-e2b"

    @app.get("/health")
    async def health():
        scheduler = app.state.scheduler
        if scheduler.stopping.is_set():
            raise HTTPException(503, "Gemma worker is unavailable")
        return {
            "status": "ok",
            "engine": scheduler.native.info,
            "scheduler": dict(scheduler.counters),
        }

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"}]}

    async def payload(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 1024 * 1024:
                raise HTTPException(413, "Request body exceeds one MiB")
        try:
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ValueError("Request must be a JSON object")
            return result
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(400, str(error)) from error

    async def complete(request, chat):
        scheduler = app.state.scheduler
        p = await payload(request)
        if p.get("model", name) != name:
            raise HTTPException(404, f"Use model {name}")
        if p.get("temperature", 0) != 0:
            raise HTTPException(400, "This Gemma endpoint currently supports greedy temperature=0")
        if p.get("n", 1) != 1 or p.get("tools") or p.get("logprobs") or p.get("echo"):
            raise HTTPException(400, "n>1, tools, logprobs and echo are not supported")
        if (
            p.get("top_p", 1) != 1
            or p.get("frequency_penalty", 0) != 0
            or p.get("presence_penalty", 0) != 0
        ):
            raise HTTPException(400, "Sampling filters and penalties are not supported")
        max_tokens = p.get("max_tokens", p.get("max_completion_tokens", 128))
        if type(max_tokens) is not int or not 1 <= max_tokens <= context_limit:
            raise HTTPException(400, "Invalid max_tokens")
        stream = p.get("stream", False)
        ignore_eos = p.get("ignore_eos", False)
        if type(stream) is not bool or type(ignore_eos) is not bool:
            raise HTTPException(400, "stream and ignore_eos must be booleans")
        stop = p.get("stop", []) or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list):
            raise HTTPException(400, "stop must be a string or array of strings")
        try:
            if chat:
                messages = p.get("messages")
                if (
                    not isinstance(messages, list)
                    or not 1 <= len(messages) <= 64
                    or any(
                        not isinstance(m, dict)
                        or m.get("role") not in {"system", "user", "assistant"}
                        or not isinstance(m.get("content"), str)
                        for m in messages
                    )
                ):
                    raise ValueError("Use one to 64 text-only chat messages")
                ids = scheduler.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
            else:
                prompt = p.get("prompt")
                if isinstance(prompt, str):
                    ids = scheduler.tokenizer.encode(prompt, add_special_tokens=True)
                elif isinstance(prompt, list):
                    ids = prompt
                else:
                    raise ValueError("prompt must be text or a single array of token IDs")
            r = scheduler.submit(ids, max_tokens, ignore_eos, stop, streaming=stream)
        except queue.Full as error:
            raise HTTPException(429, "Gemma request queue is full") from error
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        identifier = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())

        def response(result, delta=None, finish=None, token_ids=None):
            choice = {"index": 0, "finish_reason": finish}
            if p.get("return_token_ids", False):
                choice["token_ids"] = (token_ids or []) if stream else result["tokens"]
            if chat:
                choice["delta" if stream else "message"] = (
                    {"content": delta}
                    if stream
                    else {"role": "assistant", "content": result["text"]}
                )
            else:
                choice["text"] = delta if stream else result["text"]
                choice["logprobs"] = None
            value = {
                "id": identifier,
                "object": (
                    ("chat.completion.chunk" if stream else "chat.completion")
                    if chat
                    else "text_completion"
                ),
                "created": created,
                "model": name,
                "choices": [choice],
            }
            if result is not None:
                value["usage"] = {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                }
            return value

        if not stream:
            # Wake on completion instead of polling for it; look for a disconnect only while
            # still waiting.
            finished = asyncio.wrap_future(r.future)
            try:
                while not (await asyncio.wait({finished}, timeout=0.25))[0]:
                    if await request.is_disconnected():
                        r.cancelled.set()
                        raise HTTPException(499, "Client disconnected")
                result = finished.result()
                return JSONResponse(response(result, finish=result["finish_reason"]))
            except BaseException:
                r.cancelled.set()
                raise

        async def events():
            # The scheduler thread wakes this coroutine for each event, so a token costs one
            # loop callback rather than a worker-thread hand-off and a disconnect probe. Events
            # queued before the callback is installed are drained by the first pass.
            loop = asyncio.get_running_loop()
            ready = asyncio.Event()
            r.notify = lambda: loop.call_soon_threadsafe(ready.set)
            delivered = 0
            try:
                while True:
                    try:
                        event = r.events.get_nowait()
                    except queue.Empty:
                        ready.clear()
                        if not r.events.empty():
                            continue
                        try:
                            await asyncio.wait_for(ready.wait(), 0.25)
                            continue
                        except asyncio.TimeoutError:
                            pass
                        if await request.is_disconnected():
                            return
                        if r.future.done() and r.future.exception() is not None:
                            yield "data: " + json.dumps(
                                {"error": {"message": str(r.future.exception())}}
                            ) + "\n\n"
                            return
                        continue
                    delivered += 1
                    if delivered % 32 == 0 and await request.is_disconnected():
                        return
                    if "error" in event:
                        yield "data: " + json.dumps({"error": {"message": event["error"]}}) + "\n\n"
                        return
                    if "done" in event:
                        result = event["done"]
                        yield "data: " + json.dumps(
                            response(result, "", result["finish_reason"])
                        ) + "\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    yield "data: " + json.dumps(
                        response(None, event["text"], token_ids=event.get("token_ids"))
                    ) + "\n\n"
            finally:
                r.cancelled.set()

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await complete(request, False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await complete(request, True)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Serve our C++ Gemma text engine with continuous batching"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/mini_mla_gemma")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-queue", type=int, default=128)
    parser.add_argument("--full-prefill", action="store_true")
    parser.add_argument("--independent-kv", action="store_true")
    parser.add_argument("--all-batch-kv", action="store_true")
    parser.add_argument(
        "--gather-ms",
        type=float,
        default=2.0,
        help="When idle, wait this long for requests arriving together (0 admits immediately)",
    )
    args = parser.parse_args()
    import uvicorn

    app = make_app(
        args.binary.resolve(),
        args.model.resolve(),
        args.context_limit,
        args.max_batch,
        args.max_queue,
        args.full_prefill,
        not args.independent_kv,
        args.all_batch_kv,
        args.gather_ms,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
