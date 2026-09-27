"""Ratchet gate: audio-source-error-contract.

Class of bug this catches
-------------------------
A microphone read failure being disguised as a normal end-of-audio signal, and
PortAudio frame overflow being swallowed instead of reported.

The pre-fix shape (upstream ``origin/master`` 2026-09-26, eb2e039) was::

    def read_frame(self) -> bytes | None:
        if self._stream is None:
            return None
        try:
            return self._stream.read(self._frame_samples, exception_on_overflow=False)
        except Exception as e:
            logger.warning("Microphone read error: %s", e)
            return None      # error == "exhausted"

``_SourceDetector.run()`` breaks its loop on ``None``, and
``examples/basic_detection.py`` teaches the same pattern, so one transient
PortAudio error stopped wake-word detection with no exception raised, no non-zero
exit, and only a warning line to show for it. ``exception_on_overflow=False``
compounded it: dropped frames were invisible by construction, which is exactly
the bug class ``tools/collect_samples.py`` was fixed for.

Bidirectional proof required by the oracle rules:
  * FAILS on the pre-fix shape -- see ``test_detector_flags_the_pre_fix_shapes``
    (negative probes) plus the reverse run documented in the commit message.
  * PASSES on the fixed implementation -- every behaviour test below.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from violawake_sdk._exceptions import AudioCaptureError
from violawake_sdk.audio_source import (
    FRAME_BYTES,
    _MAX_CONSECUTIVE_READ_ERRORS,
    MicrophoneSource,
)

SDK_ROOT = Path(__file__).resolve().parents[2] / "src" / "violawake_sdk"
MIC_PATHS = (SDK_ROOT / "audio_source.py", SDK_ROOT / "wake_detector.py")


# --------------------------------------------------------------------------
# Detectors (pure functions over source text, so they can be run against the
# broken shapes in the negative probes below).
# --------------------------------------------------------------------------
def find_silent_overflow(source_text: str) -> list[str]:
    """Return call sites that ask PortAudio to swallow overflow."""
    violations: list[str] = []
    for node in ast.walk(ast.parse(source_text)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "read":
            continue
        for kw in node.keywords:
            if kw.arg == "exception_on_overflow" and isinstance(kw.value, ast.Constant):
                if kw.value.value is False:
                    violations.append(
                        f"line {node.lineno}: exception_on_overflow=False drops "
                        "frames silently"
                    )
    return violations


def find_error_returned_as_exhaustion(source_text: str) -> list[str]:
    """Return ``return None`` statements living inside an ``except`` handler of
    ``MicrophoneSource.read_frame`` -- the shape that fakes a device error out
    as a finished stream."""
    violations: list[str] = []
    tree = ast.parse(source_text)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name != "MicrophoneSource":
            continue
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name != "read_frame":
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Return)
                        and isinstance(sub.value, ast.Constant)
                        and sub.value.value is None
                    ):
                        violations.append(
                            f"line {sub.lineno}: read_frame returns None on a read "
                            "error (callers read None as end-of-audio and stop)"
                        )
    return violations


def mic_source_open_is_wrapped(source_text: str) -> bool:
    """True when ``MicrophoneSource.start`` converts a failed device open into
    ``AudioCaptureError`` instead of letting a bare ``OSError`` escape."""
    tree = ast.parse(source_text)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name != "MicrophoneSource":
            continue
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name != "start":
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Raise) or not node.exc:
                    continue
                name = ast.unparse(node.exc)
                if "AudioCaptureError" in name:
                    return True
        return False
    return False


# --------------------------------------------------------------------------
# Static half of the gate: the live tree must be clean, the broken shapes must
# be flagged.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("path", MIC_PATHS, ids=lambda p: p.name)
def test_no_silent_overflow_in_mic_paths(path: Path) -> None:
    assert find_silent_overflow(path.read_text(encoding="utf-8")) == []


def test_read_frame_does_not_disguise_errors_as_exhaustion() -> None:
    text = (SDK_ROOT / "audio_source.py").read_text(encoding="utf-8")
    assert find_error_returned_as_exhaustion(text) == []


def test_mic_start_wraps_failure_in_sdk_exception() -> None:
    text = (SDK_ROOT / "audio_source.py").read_text(encoding="utf-8")
    assert mic_source_open_is_wrapped(text) is True


def test_detector_flags_the_pre_fix_shapes() -> None:
    """Negative probes: the detector itself must not be vacuously green."""
    broken = '''
class MicrophoneSource:
    def start(self):
        self._stream = self._pa.open(format=1, input=True)

    def read_frame(self):
        if self._stream is None:
            return None
        try:
            return self._stream.read(320, exception_on_overflow=False)
        except Exception as e:
            logger.warning("read error: %s", e)
            return None
'''
    assert find_silent_overflow(broken) == [
        "line 10: exception_on_overflow=False drops frames silently"
    ]
    assert len(find_error_returned_as_exhaustion(broken)) == 1
    assert mic_source_open_is_wrapped(broken) is False


def test_run_loop_still_treats_none_as_end_of_source() -> None:
    """Regression guard: the fix must not be paid for by turning ``None`` into a
    raised error inside ``_SourceDetector.run()`` -- file sources legitimately
    run out."""
    text = (SDK_ROOT / "wake_detector.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if cls.name != "_SourceDetector":
            continue
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            if fn.name != "run":
                continue
            assert "if frame is None:\n                break" in ast.unparse(fn)
            return
    pytest.fail("_SourceDetector.run() disappeared from wake_detector.py")


# --------------------------------------------------------------------------
# Behaviour half of the gate: real MicrophoneSource against a fake PortAudio.
# --------------------------------------------------------------------------
class _FakeStream:
    def __init__(self, fail_times: int = 0, always_fail: bool = False) -> None:
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.read_calls: list[dict[str, object]] = []
        self.closed = False

    def read(self, num_frames: int, **kwargs: object) -> bytes:
        self.read_calls.append({"num_frames": num_frames, **kwargs})
        if self.always_fail or len(self.read_calls) <= self.fail_times:
            raise OSError(-9981, "Internal PortAudio error")
        return b"\x00" * int(num_frames) * 2

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FailingPyAudio:
    def __init__(self) -> None:
        self.terminated = False

    def open(self, **kwargs: object) -> _FakeStream:
        raise OSError(-9996, "Invalid input device")

    def terminate(self) -> None:
        self.terminated = True


def _fake_pyaudio_module(stream: _FakeStream) -> types.ModuleType:
    module = types.ModuleType("pyaudio")

    class _PyAudio:
        def __init__(self) -> None:
            self.terminated = False

        def open(self, **kwargs: object) -> _FakeStream:
            return stream

        def terminate(self) -> None:
            self.terminated = True

    module.PyAudio = _PyAudio  # type: ignore[attr-defined]
    module.paInt16 = 1  # type: ignore[attr-defined]
    return module


def _source_with(stream: _FakeStream | None) -> MicrophoneSource:
    source = MicrophoneSource.__new__(MicrophoneSource)
    source._device_index = None
    source._sample_rate = 16000
    source._frame_samples = 320
    source._pa = object()
    source._stream = stream
    return source


def test_start_failure_raises_sdk_exception_and_releases_handle() -> None:
    fake = types.ModuleType("pyaudio")
    instances: list[_FailingPyAudio] = []

    class _PyAudio:
        def __init__(self) -> None:
            self.instance = _FailingPyAudio()
            instances.append(self.instance)

        def open(self, **kwargs: object) -> _FakeStream:
            raise OSError(-9996, "Invalid input device (no default output device)")

        def terminate(self) -> None:
            self.instance.terminated = True

    fake.PyAudio = _PyAudio  # type: ignore[attr-defined]
    fake.paInt16 = 1  # type: ignore[attr-defined]
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "pyaudio", fake)
        source = MicrophoneSource(device_index=None)
        with pytest.raises(AudioCaptureError, match="Failed to open microphone"):
            source.start()
    assert instances and instances[-1].terminated, "failed open leaked the handle"


def test_transient_read_error_keeps_the_stream_alive() -> None:
    """The core of the bug: 2 failed reads used to return None and stop every
    detection loop. They must not surface as exhaustion."""
    stream = _FakeStream(fail_times=2)
    source = _source_with(stream)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("violawake_sdk.audio_source.time.sleep", lambda _s: None)
        assert source.read_frame() == b"\x00" * FRAME_BYTES


def test_persistent_read_error_raises_after_bounded_budget() -> None:
    stream = _FakeStream(always_fail=True)
    source = _source_with(stream)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("violawake_sdk.audio_source.time.sleep", lambda _s: None)
        with pytest.raises(AudioCaptureError, match="consecutive"):
            source.read_frame()
    assert len(stream.read_calls) == _MAX_CONSECUTIVE_READ_ERRORS


def test_overflow_is_not_swallowed() -> None:
    """Instrumented: the read must be issued with exception_on_overflow=True, so
    a dropped frame becomes an error the retry budget sees."""
    stream = _FakeStream(fail_times=0)
    source = _source_with(stream)
    source.read_frame()
    assert stream.read_calls[0]["exception_on_overflow"] is True


def test_unopened_source_still_returns_none() -> None:
    """Protocol preserved: None means "nothing to read", which for a microphone
    is only the not-open/stopped case."""
    assert _source_with(None).read_frame() is None
