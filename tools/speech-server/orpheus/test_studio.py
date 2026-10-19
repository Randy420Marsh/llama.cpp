#!/usr/bin/env python3
"""Unit + logic tests for the Orpheus LoRA studio (the Gradio app and its
trim/export engine).

Run with the GUI env (it needs gradio + PIL + numpy, not torch):

    /home/john/AI/sol/gui-env/bin/python tools/speech-server/orpheus/test_studio.py

Covers, in stdlib-unittest style (no pytest in that env):

  * trimtools  -- probe / peaks / 2-pass loudnorm export / clip naming / kinds
  * app logic  -- voice-folder listing + refresh, clip staging, the whole
                  trim queue flow (stage -> load -> A/B -> multi-clip export
                  -> next -> clear), length hints, time formatting
  * GUI wiring -- every `button.click(...)` / `.change(...)` handler registered
                  on the Blocks: its return value must have exactly as many
                  items as the component list it was wired to (the class of
                  bug that silently breaks buttons when a return is refactored)
"""
from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _need_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


HAVE_FFMPEG = _need_ffmpeg()


def make_media(tmp: Path) -> dict:
    """A 10 s stereo 48 kHz sine WAV and a 10 s mp4 (testsrc video + audio)."""
    wav = tmp / "tone.wav"
    mp4 = tmp / "scene.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
                    "-ar", "48000", "-ac", "2", str(wav)], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=10",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=10",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(mp4)], check=True)
    return {"wav": wav, "mp4": mp4}


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {"rate": w.getframerate(), "ch": w.getnchannels(),
                "bits": w.getsampwidth() * 8,
                "dur": w.getnframes() / w.getframerate()}


def measured_lufs(path: Path) -> float:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path),
                        "-af", "loudnorm=print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    return float(json.loads(m[-1])["input_i"])


# ---------------------------------------------------------------------------
# trimtools
# ---------------------------------------------------------------------------

class TrimToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAVE_FFMPEG:
            cls.skipReason = "ffmpeg/ffprobe not on PATH"
            raise unittest.SkipTest(cls.skipReason)
        cls.tmp = Path(tempfile.mkdtemp(prefix="orpheus-test-"))
        cls.media = make_media(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_probe_video(self):
        import trimtools
        info = trimtools.probe(self.media["mp4"])
        self.assertAlmostEqual(info["duration"], 10.0, delta=0.5)
        self.assertTrue(info["has_video"] and info["has_audio"])
        self.assertEqual((info["width"], info["height"]), (320, 240))

    def test_probe_audio(self):
        import trimtools
        info = trimtools.probe(self.media["wav"])
        self.assertTrue(info["has_audio"])
        self.assertFalse(info["has_video"])
        self.assertAlmostEqual(info["duration"], 10.0, delta=0.5)

    def test_peaks(self):
        import trimtools
        pk = trimtools.peaks(self.media["wav"], 200)
        self.assertEqual(len(pk), 200)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in pk))
        self.assertGreater(pk[100], 0.1)   # a 440 Hz sine is far from silent
        silence = self.tmp / "silence.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
                        "-t", "1", str(silence)], check=True)
        self.assertTrue(all(v == 0.0 for v in trimtools.peaks(silence, 10)))

    def test_export_full_and_format(self):
        import trimtools
        out = self.tmp / "out" / "0001.wav"
        r = trimtools.export_clip(self.media["wav"], out)
        self.assertAlmostEqual(r["duration"], 10.0, delta=0.5)
        wi = wav_info(out)
        self.assertEqual((wi["rate"], wi["ch"], wi["bits"]), (24000, 1, 16))
        self.assertAlmostEqual(measured_lufs(out), -23.0, delta=1.0)

    def test_export_trimmed(self):
        import trimtools
        out = self.tmp / "out" / "0002.wav"
        r = trimtools.export_clip(self.media["mp4"], out, 3.0, 9.0)   # video source
        self.assertAlmostEqual(r["duration"], 6.0, delta=0.5)
        self.assertEqual(wav_info(out)["rate"], 24000)

    def test_next_clip_name(self):
        import trimtools
        d = self.tmp / "voice"
        self.assertEqual(trimtools.next_clip_name(d).name, "0001.wav")
        (d / "0003.wav").write_bytes(b"x")
        (d / "0010.wav").write_bytes(b"x")
        (d / "reference.wav").write_bytes(b"x")     # not a NNNN clip
        self.assertEqual(trimtools.next_clip_name(d).name, "0011.wav")

    def test_kind_of(self):
        import trimtools
        self.assertEqual(trimtools.kind_of(Path("a.mp4")), "video")
        self.assertEqual(trimtools.kind_of(Path("a.WAV")), "audio")
        self.assertEqual(trimtools.kind_of(Path("a.webm")), "video")

    def test_ensure_ffmpeg(self):
        import trimtools
        self.assertIsNone(trimtools.ensure_ffmpeg())


# ---------------------------------------------------------------------------
# app logic (imported with the voices/orpheus dirs pointed at temp locations)
# ---------------------------------------------------------------------------

class AppLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="orpheus-app-"))
        os.environ["ORPHEUS_OUT"] = str(cls.tmp / "orpheus")
        os.environ["ORPHEUS_VOICES"] = str(cls.tmp / "voices")
        os.environ.setdefault("ORPHEUS_PORT", "7999")
        import importlib
        fresh = "app" not in sys.modules
        import app
        if not fresh:
            app = importlib.reload(app)
        cls.app = app
        cls.media = make_media(cls.tmp)
        (cls.app.VOICES / "alice").mkdir(parents=True)
        (cls.app.VOICES / "alice" / "reference.wav").write_bytes(b"")
        (cls.app.VOICES / "alice" / "transcript.txt").write_text(
            "hello world this is a long transcript for testing")
        (cls.app.VOICES / "empty").mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.app._TRIM_QUEUE.clear()
        for p in (self.app.ORPHEUS_DIR / "staging").glob("*") \
                if (self.app.ORPHEUS_DIR / "staging").is_dir() else []:
            p.unlink()

    def _files(self, *paths):
        return [type("F", (), {"name": str(p)}) for p in paths]

    # --- voice folder listing -------------------------------------------

    def test_voice_names_lists_every_folder(self):
        names = self.app._voice_names()
        self.assertIn("alice", names)      # has a reference
        self.assertIn("empty", names)      # empty folder: still listed
        new = self.app.VOICES / "Manual"    # created while the studio runs
        new.mkdir()
        self.assertIn("Manual", self.app._voice_names())

    def test_voice_choices_require_reference(self):
        choices = {v for _lbl, v in self.app._voice_choices()}
        self.assertTrue(any("alice" in c for c in choices))
        self.assertFalse(any("empty" in c for c in choices))
        labels = [lbl for lbl, _v in self.app._voice_choices()]
        self.assertTrue(any("hello world this is" in l for l in labels))

    def test_refresh_voices_returns_dict_and_keeps_selection(self):
        d = self.app._refresh_voices("")
        self.assertIsInstance(d, dict)
        self.assertIn("choices", d) and self.assertIn("value", d)
        self.assertEqual(d["value"], sorted(d["choices"])[0])
        d2 = self.app._refresh_voices("alice")
        self.assertEqual(d2["value"], "alice")
        d3 = self.app._refresh_voices("typed-new-name")   # not created yet
        self.assertEqual(d3["value"], "typed-new-name")

    # --- time / length helpers ------------------------------------------

    def test_time_format(self):
        self.assertEqual(self.app._trim_fmt_t(0), "0:00")
        self.assertEqual(self.app._trim_fmt_t(75), "1:15")
        self.assertEqual(self.app._trim_fmt_t(3671), "1:01:11")

    def test_length_hint(self):
        self.assertIn("in the 6-15 s", self.app._trim_len_hint(0, 10))
        self.assertIn("outside", self.app._trim_len_hint(0, 3))

    # --- dataset clip staging --------------------------------------------

    def test_stage_clips(self):
        folder, msg = self.app.stage_clips(
            self._files(self.media["wav"]), str(self.tmp / "clips"))
        self.assertTrue(Path(folder, "tone.wav").is_file())
        self.assertIn("1 clip(s) staged", msg)
        folder2, msg2 = self.app.stage_clips(
            self._files(self.media["wav"]), str(self.tmp / "clips"))
        self.assertIn("skipped", msg2)                      # same name again
        _, msg3 = self.app.stage_clips(None, str(self.tmp / "clips"))
        self.assertEqual(msg3, "No files dropped yet.")

    # --- the full trim queue flow ----------------------------------------

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
    def test_trim_queue_full_flow(self):
        app = self.app
        # stage a video + a wav with trim ON -> both queued, video gets m4a
        rows, msg = app.trim_stage(
            self._files(self.media["mp4"], self.media["wav"]), "alice", True)[:2]
        self.assertIn("2 item(s) queued", msg)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[5] == "pending" for r in rows))
        self.assertTrue(any(r[2] == "video" for r in rows))
        self.assertTrue(any(r[2] == "audio" for r in rows))
        it = app._trim_pending()
        self.assertIsNotNone(it)
        # the video's audio track was extracted for the synced player
        if it["kind"] == "video":
            self.assertTrue(Path(it["audio"]).is_file(), "m4a extraction failed")
        if it["kind"] == "audio":
            it = next(x for x in app._TRIM_QUEUE if x["kind"] == "video")
            self.assertTrue(Path(it["audio"]).is_file())

        # Load next: 9 values = status + (video, audio, wave, info, A, B, len, cur)
        out = app.trim_load()
        self.assertEqual(len(out), 9)
        self.assertEqual(out[5], 0.0)                        # A
        self.assertAlmostEqual(out[6], 10.0, delta=0.5)      # B = min(REC_MAX, dur=10)
        cur = out[-1]

        # A/B marks redraw the waveform + hint
        img, hint = app.trim_marks(cur, 2.0, 8.0)
        self.assertEqual(img.size, (900, 160))
        self.assertIn("in the 6-15 s", hint)

        # two clips from the SAME long source (multi-clip), into a NEW voice
        rows, msg = app.trim_export(2.0, 8.0, "newvoice", cur)[:2]
        self.assertIn("0001.wav", msg)
        self.assertTrue((app.VOICES / "newvoice" / "0001.wav").is_file())
        rows, msg = app.trim_export(5.0, 11.0, "newvoice", cur)[:2]
        self.assertIn("0002.wav", msg)
        self.assertIn("2 clip(s) from this item", msg)
        item = app._trim_find(cur)
        self.assertEqual(item["exports"], 2)
        self.assertEqual(item["status"], "done")

        # Next item marks it processed and advances to the other pending file
        rows, note, *prev = app.trim_next(cur)
        self.assertIn("processed", note)
        self.assertEqual(len(prev), 8)
        nxt = app._trim_pending()
        self.assertIsNotNone(nxt)
        self.assertNotEqual(nxt["path"], cur)

        # validation paths
        rows, msg = app.trim_export(0.0, 0.1, "newvoice", nxt["path"])[:2]
        self.assertIn("0.3 s after A", msg)
        rows, msg = app.trim_export(0.0, 5.0, "newvoice", "no-such-path")[:2]
        self.assertIn("Load an item first", msg)

        # skip the remaining pending item without exporting
        app.trim_load()
        rows, note = app.trim_next(nxt["path"])[:2]
        self.assertIn("skipped", note)
        self.assertEqual(app._trim_pending(), None)

        # export into an existing voice continues the numbering
        rows, msg = app.trim_stage(
            self._files(self.media["wav"]), "newvoice", False)[:2]
        self.assertIn("0003.wav", msg)

        # Clear wipes queue + staging files
        staged = list((app.ORPHEUS_DIR / "staging").iterdir())
        self.assertTrue(staged)
        out = app.trim_clear()
        self.assertEqual(out[0], [])
        self.assertIn("Queue cleared", out[1])
        self.assertEqual(len(out), 10)
        self.assertEqual(list((app.ORPHEUS_DIR / "staging").iterdir()), [])

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
    def test_trim_stage_no_files(self):
        rows, msg = self.app.trim_stage(self._files(self.tmp / "missing.mp4"),
                                        "alice", True)[:2]
        self.assertIn("No readable", msg)

    def test_load_empty_queue(self):
        out = self.app.trim_load()
        self.assertEqual(len(out), 9)
        self.assertEqual(out[0], "No pending items left.")

    def test_next_without_item(self):
        out = self.app.trim_next("")
        self.assertEqual(len(out), 10)
        self.assertEqual(out[1], "No current item.")


# ---------------------------------------------------------------------------
# GUI wiring: every handler's return width must match its wired components
# ---------------------------------------------------------------------------

class GuiWiringTests(unittest.TestCase):
    """Reproduces the class of bug where a handler returns N values but its
    `.click(...)` is wired to M components (M != N crashes at click time).
    Runs each registered handler with dummy inputs and compares."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="orpheus-wire-"))
        os.environ["ORPHEUS_OUT"] = str(cls.tmp / "orpheus")
        os.environ["ORPHEUS_VOICES"] = str(cls.tmp / "voices")
        os.environ.setdefault("ORPHEUS_PORT", "7999")
        import importlib
        fresh = "app" not in sys.modules
        import app
        if not fresh:
            app = importlib.reload(app)
        cls.app = app
        # make the job generators cheap + side-effect-free
        def fake_stream(cmd, cwd=".", env=None):
            yield ("wiring-test", "wiring-test")
        cls._real_stream = cls.app.stream
        cls.app.stream = fake_stream
        import subprocess as sp

        class _FakeProc:
            stdout = iter(["fake line\n"])
            def wait(self):
                return 0
        cls._real_popen = cls.app.subprocess.Popen
        cls.app.subprocess.Popen = lambda *a, **k: _FakeProc()
        cls.dummy = cls.tmp / "dummy.wav"
        cls.dummy.write_bytes(b"")

    @classmethod
    def tearDownClass(cls):
        cls.app.stream = cls._real_stream
        cls.app.subprocess.Popen = cls._real_popen
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_every_handler_matches_its_outputs(self):
        demo = self.app.demo
        checked = []
        for fn in (demo.fns.values() if isinstance(demo.fns, dict) else demo.fns):
            f = fn.fn
            nin, nout = len(fn.inputs), len(fn.outputs)
            # dummy inputs: File-like objects where a drop-zone feeds in,
            # plain strings elsewhere (some handlers do Path(x) / float(x))
            def _f():
                return type("F", (), {"name": str(self.dummy)})()
            if f.__name__ == "stage_clips":
                args = [[_f()], ""]
            elif f.__name__ == "trim_stage":
                args = [[_f()], "alice", True]
            elif f.__name__ == "trim_marks":
                args = ["x", 2.0, 8.0]
            else:
                args = ["x" for _ in range(nin)]
            if inspect.isgeneratorfunction(f):
                gen = f(*args)
                first = next(gen)
                n = len(first) if isinstance(first, tuple) else 1
            else:
                r = f(*args)
                n = 1 if isinstance(r, dict) else len(r)
            self.assertEqual(
                n, nout,
                f"{f.__name__}: returns {n} value(s) but is wired to "
                f"{nout} component(s)")
            checked.append(f.__name__)
        # the important ones really got exercised:
        for name in ("trim_stage", "trim_load", "trim_export", "trim_next",
                     "trim_clear", "trim_marks", "_refresh_voices", "status",
                     "stage_clips", "job_dataset", "job_transcribe",
                     "job_train", "job_publish", "job_synth"):
            self.assertIn(name, checked, f"{name} is no longer registered?")


if __name__ == "__main__":
    unittest.main(verbosity=2)
