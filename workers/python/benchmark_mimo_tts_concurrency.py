#!/usr/bin/env python3
"""Benchmark the provider-side MiMo TTS concurrency limit without exposing credentials.

This is intentionally a raw HTTP benchmark: it bypasses the application's
serial MiMo gate so that it can measure the provider's response to a burst.
It first prepares one reusable voice-clone reference, then sends one short
voice-clone request per worker simultaneously.  It starts at four workers and
falls back to lower levels only after a failed/limited wave.

No audio is written to disk.  The Markdown report contains only safe metadata:
HTTP status, elapsed time, selected rate-limit headers, request sizes, and any
provider supplied usage/cost object.  It never writes the API key or audio data.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from audiobook_worker.tts import (  # noqa: E402
    _MIMO_ENDPOINT,
    _MIMO_REFERENCE_TEXT,
    _MIMO_VOICE_CLONE_MODEL_ID,
    _MIMO_VOICE_DESIGN_MODEL_ID,
    _MIMO_VOICE_DESIGNS,
    _load_mimo_api_key,
)


SAFE_RESPONSE_HEADERS = {
    "retry-after",
    "x-request-id",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
}


@dataclass
class RequestResult:
    request_id: int
    status: int | None
    elapsed_seconds: float
    response_audio_bytes: int | None
    request_bytes: int
    text_characters: int
    provider_usage: dict[str, Any] | None
    provider_cost: Any
    rate_limit_headers: dict[str, str]
    error_kind: str | None
    error_message: str | None
    started_offset_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error_kind is None


def _safe_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    result: dict[str, str] = {}
    for name in SAFE_RESPONSE_HEADERS:
        value = headers.get(name)
        if value:
            result[name] = str(value)
    return result


def _error_message(raw: bytes) -> str | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")[:240]
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("type") or error.get("code")
            return str(message)[:240] if message is not None else "provider returned an error"
        message = parsed.get("message")
        if message is not None:
            return str(message)[:240]
    return "provider returned an error"


def _post_json(api_key: str, payload: dict[str, Any], timeout: float) -> RequestResult:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    text = ""
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        candidate = messages[-1]
        if isinstance(candidate, dict):
            text = str(candidate.get("content") or "")
    request = urllib.request.Request(
        _MIMO_ENDPOINT,
        data=encoded,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - started
            parsed = json.loads(raw.decode("utf-8"))
            audio_data = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("audio", {})
                .get("data")
            )
            if not isinstance(audio_data, str):
                raise ValueError("MiMo response did not contain choices[0].message.audio.data")
            try:
                audio_bytes = len(base64.b64decode(audio_data, validate=True))
            except (ValueError, TypeError) as error:
                raise ValueError("MiMo response audio is not valid Base64") from error
            usage = parsed.get("usage")
            cost = parsed.get("cost")
            return RequestResult(
                request_id=-1,
                status=int(getattr(response, "status", 200)),
                elapsed_seconds=elapsed,
                response_audio_bytes=audio_bytes,
                request_bytes=len(encoded),
                text_characters=len(text),
                provider_usage=usage if isinstance(usage, dict) else None,
                provider_cost=cost,
                rate_limit_headers=_safe_headers(response.headers),
                error_kind=None,
                error_message=None,
            )
    except urllib.error.HTTPError as error:
        elapsed = time.perf_counter() - started
        return RequestResult(
            request_id=-1,
            status=error.code,
            elapsed_seconds=elapsed,
            response_audio_bytes=None,
            request_bytes=len(encoded),
            text_characters=len(text),
            provider_usage=None,
            provider_cost=None,
            rate_limit_headers=_safe_headers(error.headers),
            error_kind="http_error",
            error_message=_error_message(error.read()),
        )
    except Exception as error:  # network and decoding failures are benchmark results
        elapsed = time.perf_counter() - started
        return RequestResult(
            request_id=-1,
            status=None,
            elapsed_seconds=elapsed,
            response_audio_bytes=None,
            request_bytes=len(encoded),
            text_characters=len(text),
            provider_usage=None,
            provider_cost=None,
            rate_limit_headers={},
            error_kind=type(error).__name__,
            error_message=str(error)[:240],
        )


def _reference_payload() -> dict[str, Any]:
    description = _MIMO_VOICE_DESIGNS["narrator_female"]
    return {
        "model": _MIMO_VOICE_DESIGN_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{description}\n\n"
                    "指导：生成一段自然、稳定、克制的基础音色参考样本。"
                    "只建立固定音色，不加入夸张表演或后期效果。"
                ),
            },
            {"role": "assistant", "content": _MIMO_REFERENCE_TEXT},
        ],
        "audio": {"format": "wav", "optimize_text_preview": False},
    }


def _clone_payload(reference_audio: str, request_number: int) -> dict[str, Any]:
    text = f"这是 MiMo 并发测试第 {request_number} 条短句。"
    return {
        "model": _MIMO_VOICE_CLONE_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": (
                    "角色：严格复用参考音频中的同一位说话者。保持固定音色、自然语速和清晰咬字。"
                    "场景：中性技术测试。指导：平静自然朗读，不添加情绪化表演。"
                ),
            },
            {"role": "assistant", "content": text},
        ],
        "audio": {"format": "wav", "voice": reference_audio},
    }


def _extract_audio_data(payload: dict[str, Any], api_key: str, timeout: float) -> tuple[str | None, RequestResult]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        _MIMO_ENDPOINT,
        data=encoded,
        headers={"api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed = time.perf_counter() - started
            parsed = json.loads(raw.decode("utf-8"))
            audio_data = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("audio", {})
                .get("data")
            )
            if not isinstance(audio_data, str):
                raise ValueError("MiMo reference response did not contain audio data")
            audio_bytes = len(base64.b64decode(audio_data, validate=True))
            voice_reference = (
                audio_data
                if audio_data.startswith("data:audio/")
                else f"data:audio/wav;base64,{audio_data}"
            )
            return voice_reference, RequestResult(
                request_id=0,
                status=int(getattr(response, "status", 200)),
                elapsed_seconds=elapsed,
                response_audio_bytes=audio_bytes,
                request_bytes=len(encoded),
                text_characters=len(_MIMO_REFERENCE_TEXT),
                provider_usage=parsed.get("usage") if isinstance(parsed.get("usage"), dict) else None,
                provider_cost=parsed.get("cost"),
                rate_limit_headers=_safe_headers(response.headers),
                error_kind=None,
                error_message=None,
            )
    except urllib.error.HTTPError as error:
        return None, RequestResult(
            request_id=0,
            status=error.code,
            elapsed_seconds=time.perf_counter() - started,
            response_audio_bytes=None,
            request_bytes=len(encoded),
            text_characters=len(_MIMO_REFERENCE_TEXT),
            provider_usage=None,
            provider_cost=None,
            rate_limit_headers=_safe_headers(error.headers),
            error_kind="http_error",
            error_message=_error_message(error.read()),
        )
    except Exception as error:
        return None, RequestResult(
            request_id=0,
            status=None,
            elapsed_seconds=time.perf_counter() - started,
            response_audio_bytes=None,
            request_bytes=len(encoded),
            text_characters=len(_MIMO_REFERENCE_TEXT),
            provider_usage=None,
            provider_cost=None,
            rate_limit_headers={},
            error_kind=type(error).__name__,
            error_message=str(error)[:240],
        )


def _run_wave(
    api_key: str,
    reference_audio: str,
    concurrency: int,
    timeout: float,
    start_interval_seconds: float,
) -> dict[str, Any]:
    first_start_at = time.monotonic() + 0.2

    def worker(request_id: int) -> RequestResult:
        scheduled_start = first_start_at + (request_id - 1) * start_interval_seconds
        wait_seconds = scheduled_start - time.monotonic()
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        actual_start = time.monotonic()
        result = _post_json(api_key, _clone_payload(reference_audio, request_id), timeout)
        result.request_id = request_id
        result.started_offset_seconds = round(actual_start - first_start_at, 3)
        return result

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, request_id) for request_id in range(1, concurrency + 1)]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item.request_id)
    return {
        "concurrency": concurrency,
        "startIntervalSeconds": start_interval_seconds,
        "waveElapsedSeconds": round(elapsed, 3),
        "successCount": sum(item.ok for item in results),
        "http429Count": sum(item.status == 429 for item in results),
        "results": [asdict(item) for item in results],
    }


def _retry_after_seconds(wave: dict[str, Any], default: float) -> float:
    retry_values: list[float] = []
    for result in wave["results"]:
        value = result.get("rate_limit_headers", {}).get("retry-after")
        if value:
            try:
                retry_values.append(float(value))
            except ValueError:
                pass
    return max(retry_values, default) if retry_values else default


def _numeric_usage_totals(results: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for result in results:
        usage = result.get("provider_usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MiMo TTS 并发压测报告",
        "",
        f"- 执行时间：{report['executedAt']}",
        f"- 端点：`{report['endpoint']}`",
        "- 模型：`mimo-v2.5-tts-voiceclone`（先用 voicedesign 生成一个共享参考音频）",
        "- 方法：原始 HTTP 并发波次；绕过应用内串行限流器，仅用于测量供应商接口的并发响应。",
        "- 重试：关闭。每个请求只发送一次，避免把重试流量误计入并发容量。",
        f"- 参考音频后静置：{report['settleAfterReferenceSeconds']:.1f} 秒。",
        f"- 请求启动间隔：{report['startIntervalSeconds']:.3f} 秒。",
        "- 敏感信息：未记录 API Key、参考音频或返回音频内容。",
        "",
        "## 参考音频预热",
        "",
        f"- HTTP 状态：{report['reference']['status']}",
        f"- 耗时：{report['reference']['elapsed_seconds']:.3f} 秒",
        f"- 音频字节数：{report['reference']['response_audio_bytes'] or '—'}",
        f"- Provider usage：`{json.dumps(report['reference']['provider_usage'], ensure_ascii=False) if report['reference']['provider_usage'] is not None else '未返回'}`",
        f"- Provider cost：`{json.dumps(report['reference']['provider_cost'], ensure_ascii=False) if report['reference']['provider_cost'] is not None else '未返回'}`",
        "",
        "## 并发结果",
        "",
        "| 并发 | 启动间隔 | 成功 | 429 | 波次耗时 | 结论 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for wave in report["waves"]:
        success = wave["successCount"]
        concurrency = wave["concurrency"]
        outcome = "通过" if success == concurrency else "失败/受限"
        lines.append(
            f"| {concurrency} | {wave['startIntervalSeconds']:.3f}s | {success}/{concurrency} | {wave['http429Count']} | "
            f"{wave['waveElapsedSeconds']:.3f}s | {outcome} |"
        )

    lines += ["", "## 每请求明细", ""]
    for wave in report["waves"]:
        lines += [f"### 并发 {wave['concurrency']}", ""]
        lines += [
            "| 请求 | 实际启动偏移 | HTTP | 耗时 | 返回音频字节 | 请求字节 | 文本字符 | usage | cost | 错误 |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
        for item in wave["results"]:
            usage = json.dumps(item["provider_usage"], ensure_ascii=False) if item["provider_usage"] is not None else "—"
            cost = json.dumps(item["provider_cost"], ensure_ascii=False) if item["provider_cost"] is not None else "—"
            error = item["error_message"] or "—"
            lines.append(
                f"| {item['request_id']} | {item['started_offset_seconds'] if item['started_offset_seconds'] is not None else '—'}s | {item['status'] or '—'} | {item['elapsed_seconds']:.3f}s | "
                f"{item['response_audio_bytes'] or '—'} | {item['request_bytes']} | {item['text_characters']} | "
                f"`{usage}` | `{cost}` | {error} |"
            )
        totals = _numeric_usage_totals(wave["results"])
        lines += [
            "",
            f"Provider usage 数值汇总：`{json.dumps(totals, ensure_ascii=False) if totals else 'API 未返回 usage，无法可靠换算 token 或金额'}`",
            "",
        ]

    passed = report.get("highestPassingConcurrency")
    lines += [
        "## 结论",
        "",
        (
            f"- 在本次单波次测试中，应用并发上限 4 以内的最高通过值：**{passed}**。"
            if passed is not None
            else "- 没有找到完全通过的并发值；请查看 HTTP 状态与供应商限流头。"
        ),
        "- `usage` / `cost` 仅记录供应商响应实际返回的字段；若显示“未返回”，不能用本地字符数可靠推导 MiMo 的计费 token 或金额。",
        "- 单次波次只能反映该时刻的供应商配额；生产配置仍应保留限流与 429 退避。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="Markdown report path")
    parser.add_argument("--start", type=int, default=4, help="Initial parallel request count (default: 4)")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds")
    parser.add_argument(
        "--cooldown-after-429",
        type=float,
        default=65.0,
        help="Fallback cooldown before lowering concurrency when Retry-After is absent",
    )
    parser.add_argument(
        "--settle-after-reference",
        type=float,
        default=60.0,
        help="Wait after voice-reference preparation before starting the first clone wave",
    )
    parser.add_argument(
        "--start-interval",
        type=float,
        default=0.0,
        help="Seconds between starts within one wave; 0 starts all requests together",
    )
    parser.add_argument(
        "--only-level",
        action="store_true",
        help="Run only --start instead of automatically testing lower levels after a failure",
    )
    args = parser.parse_args()
    if args.start < 1 or args.start > 4:
        parser.error("--start must be in the application's supported range 1..4")
    if args.start_interval < 0:
        parser.error("--start-interval must be non-negative")

    api_key = _load_mimo_api_key()
    if not api_key:
        print("MiMo API key is not available from the current Keychain configuration.", file=sys.stderr)
        return 2

    reference_audio, reference = _extract_audio_data(_reference_payload(), api_key, args.timeout)
    report: dict[str, Any] = {
        "executedAt": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "endpoint": _MIMO_ENDPOINT,
        "reference": asdict(reference),
        "waves": [],
        "highestPassingConcurrency": None,
        "settleAfterReferenceSeconds": args.settle_after_reference,
        "startIntervalSeconds": args.start_interval,
    }
    if reference_audio is None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_markdown(report), encoding="utf-8")
        print(f"Reference preparation failed; report written to {args.report}", file=sys.stderr)
        return 1

    if args.settle_after_reference > 0:
        print(
            f"Waiting {args.settle_after_reference:.1f}s after reference preparation "
            "before the first clone wave.",
            flush=True,
        )
        time.sleep(args.settle_after_reference)

    levels = [args.start] if args.only_level else range(args.start, 0, -1)
    for level in levels:
        print(
            f"Running MiMo voiceclone wave at concurrency={level}, "
            f"start interval={args.start_interval:.3f}s...",
            flush=True,
        )
        wave = _run_wave(
            api_key,
            reference_audio,
            level,
            args.timeout,
            args.start_interval,
        )
        report["waves"].append(wave)
        if wave["successCount"] == level:
            report["highestPassingConcurrency"] = level
            break
        if wave["http429Count"] and not args.only_level:
            delay = _retry_after_seconds(wave, args.cooldown_after_429)
            print(f"Received 429; cooling down for {delay:.1f}s before the next lower wave.", flush=True)
            time.sleep(delay)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(report), encoding="utf-8")
    print(f"Report written to {args.report}")
    return 0 if report["highestPassingConcurrency"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
