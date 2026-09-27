"""Detect wake words from the microphone using a model YOU trained.

The other examples load a registry model (``WakeDetector(model="temporal_cnn")``).
This one loads a custom ``.onnx`` exported by ``violawake-train`` and drives it
from ``MicrophoneSource``, which is the path an integrator takes after training:

    violawake-collect --word stop --output data/stop/positives --count 150
    violawake-train --word stop --positives data/stop/positives \
        --negatives data/stop/negatives --output models/stop.onnx
    python examples/mic_detection_with_trained_model.py --model models/stop.onnx

Audio contract (asserted by the SDK itself, do not "tune" these):
16 kHz mono, 20 ms frames = 320 samples = 640 bytes of little-endian int16 PCM.

Requires: pip install "violawake[audio]"
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from violawake_sdk import WakeDetector
from violawake_sdk._exceptions import AudioCaptureError, ViolaWakeError
from violawake_sdk.audio_source import FileSource, MicrophoneSource
from violawake_sdk.wake_detector import FRAME_SAMPLES, SAMPLE_RATE


def _list_devices() -> int:
    try:
        import pyaudio
    except ImportError:
        print("pyaudio is required: pip install 'violawake[audio]'")
        return 1
    pa = pyaudio.PyAudio()
    try:
        print(f"{'index':>5}  {'channels':>8}  name")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxInputChannels", 0)) > 0:
                print(f"{i:>5}  {int(info['maxInputChannels']):>8}  {info['name']}")
    finally:
        pa.terminate()
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microphone wake-word detection with a trained model.",
    )
    parser.add_argument(
        "--model",
        default="models/stop.onnx",
        help="Path to a trained .onnx, or a registry name like 'temporal_cnn'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Detection threshold in [0, 1] (default 0.80, the trained value).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.0,
        help="Minimum seconds between two detections (default 1.0).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="PyAudio input device index (default: system default device).",
    )
    parser.add_argument(
        "--no-meter",
        action="store_true",
        help="Do not print the live score meter; only print detections.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "--wav",
        default=None,
        metavar="PATH",
        help="Offline mode: run the same detection loop over a 16 kHz mono "
        "16-bit WAV instead of the microphone (handy for CI and for checking "
        "a recording without opening the device).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_devices:
        return _list_devices()

    # 1. Load the trained model. Any path works; the SDK reads the ONNX and the
    #    pinned OpenWakeWord backbone it was trained against (ADR-002).
    try:
        detector = WakeDetector(
            model=args.model,
            threshold=args.threshold,
            cooldown_s=args.cooldown,
        )
    except (FileNotFoundError, ViolaWakeError) as exc:
        print(f"Could not load model {args.model!r}: {exc}")
        print("Pass --model path/to/your.onnx (see violawake-train --output).")
        return 1

    # 2. Open the microphone through the audio-source layer (or a file, for
    #    offline checks -- both satisfy the same AudioSource protocol).
    if args.wav:
        source: MicrophoneSource | FileSource = FileSource(args.wav)
    else:
        source = MicrophoneSource(device_index=args.device)

    print(f"model     : {args.model}")
    print(f"source    : {'file ' + args.wav if args.wav else 'microphone'}")
    print(f"audio     : {SAMPLE_RATE} Hz mono, {FRAME_SAMPLES} samples / 20 ms frame")
    print(f"threshold : {args.threshold:.2f}   cooldown: {args.cooldown:.1f}s")
    print("Say the wake word. Ctrl-C to stop.\n")

    detections = 0
    last_meter = 0.0
    started = time.monotonic()
    try:
        source.start()
        while True:
            frame = source.read_frame()
            if frame is None:
                print("\nAudio source exhausted.")
                break

            # detect() runs inference once and applies the 4-gate decision
            # policy (RMS floor / clipping / music / debounce). Read the score
            # AFTER it from last_scores -- calling process() as well would run
            # the model a second time on the same frame.
            hit = detector.detect(frame)
            score = detector.last_scores[-1] if detector.last_scores else 0.0

            if hit:
                detections += 1
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{stamp}] WAKE!  score={score:.3f}  (detection #{detections})")
                last_meter = time.monotonic()  # let the WAKE line stand

            now = time.monotonic()
            if not args.no_meter and (now - last_meter) >= 0.2:
                last_meter = now
                bar = "#" * int(score * 30)
                marker = "*" if score >= args.threshold else " "
                print(f"\r  score {score:.3f} {marker}|{bar:<30}|  ", end="", flush=True)

            if args.duration is not None and (now - started) >= args.duration:
                print("\nDuration reached.")
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (AudioCaptureError, ImportError, OSError, FileNotFoundError) as exc:
        # MicrophoneSource.start() lets the raw PortAudio OSError escape (only
        # WakeDetector.stream_mic wraps it into AudioCaptureError), so catch
        # both shapes here rather than printing a traceback at the user.
        print(f"\nAudio source unavailable: {exc}")
        print("Check the device: python examples/"
              "mic_detection_with_trained_model.py --list-devices")
        return 1
    finally:
        source.stop()
        detector.close()

    print(f"\nTotal detections: {detections}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
