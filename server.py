# Imports 
import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import av
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from groq import AsyncGroq
from faster_whisper import WhisperModel


# Logging 
# Root logger at DEBUG; third-party libraries silenced to WARNING.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

for noisy in ("httpx", "httpcore", "faster_whisper", "av", "groq"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)


## Configuration 
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
RESPONSE_MODEL = "llama-3.3-70b-versatile"

# "groq"  - Groq Whisper Large V3 Turbo (~300ms, best for accented English)
# "local" - faster-whisper on CPU (~2-4s)
TRANSCRIPTION_BACKEND = "groq"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# Local Whisper settings - only loaded when TRANSCRIPTION_BACKEND="local" (fallback)
WHISPER_MODEL_SIZE = "medium"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

VAD_SILENCE_THRESHOLD_S = 1.0       # seconds of silence that ends a turn
                                    # Lowered from 1.5s: noise floor is clean (VAD ~0.000–0.018)
                                    # and 0.5s saved per turn compounds significantly over a session.
                                    # For Condition C, reflective utterances use a dynamic threshold
                                    # of 3.0s regardless of this value.

# Speculative transcription: start a Groq call this many seconds into silence,
# before the turn threshold fires, so the result is ready (or nearly so) by
# the time fire_turn() would otherwise start transcription from scratch.
# Set to ~half the silence threshold so the call overlaps the wait window.
# Only applies to Condition A; Conditions B/C already prefetch via their own path.
SPECULATIVE_TRANSCRIBE_AT_S: float = 0.50

MAX_SPEECH_CHUNKS = 120             # hard cap at ~36s; fires turn regardless of silence
SPEECH_PROB_THRESHOLD = 0.85        # Silero probability above which a window is speech
CHUNK_MS = 300                      # must match MediaRecorder timeslice on the client
SAMPLE_RATE = 16_000
VAD_WINDOW = 512                    # Silero requires exactly 512 samples @ 16 kHz
PRE_SPEECH_PAD = 8                  # silent chunks prepended before first voiced chunk (~2.4 s)
MAX_SILENCE_S: float = 4.0   # hard cap for B/C — fire unconditionally after this long
                             # (transcription + EOS adds ~1.5s on top, so worst-case
                             # response time is ~5.5s, acceptable for deliberative speech)

# ── Condition flag ─────────────────────────────────────────────────────────────
# "A" — VAD only (baseline): fires on silence threshold alone
# "B" — VAD + EOS scorer: syntactic completeness gates the turn decision
# "C" — Full ensemble: task classifier + LLM scorer + EOS + dynamic threshold
CONDITION: str = "A"

# ── Dynamic silence thresholds per utterance type (Condition C only) ───────────
# Transactional utterances get a short window; reflective get more space.
DYNAMIC_THRESHOLDS: dict[str, float] = {
    "transactional": 0.8,
    "informational":  1.8,
    "reflective":     3.0,
    "unknown":        VAD_SILENCE_THRESHOLD_S,  # fallback to default
}

# ── Ensemble weights (Condition C only) ───────────────────────────────────────
# Combined score = EOS_WEIGHT * eos_score + LLM_WEIGHT * llm_score
# Scores are each in [0, 1]; combined score >= ENSEMBLE_THRESHOLD fires the turn.
EOS_WEIGHT:         float = 0.2
LLM_WEIGHT:         float = 0.8
ENSEMBLE_THRESHOLD: float = 0.65

# ── High-confidence early-fire override (Condition C only) ────────────────────
# For task types whose dynamic silence threshold exceeds VAD_SILENCE_THRESHOLD_S
# (informational: 1.8s, reflective: 3.0s), the ensemble can fire early when the
# combined score is unambiguously high — avoiding the full dynamic wait on
# utterances that are clearly complete (e.g. reflective with EOS=1.0, LLM=0.9).
#
# Two-tier behaviour:
#   combined >= HIGH_CONFIDENCE_THRESHOLD  → fire immediately (after EARLY_FIRE_MIN_SILENCE_S)
#   combined >= ENSEMBLE_THRESHOLD         → fire only once dynamic threshold is met
#   combined <  ENSEMBLE_THRESHOLD         → hold
#
# EARLY_FIRE_MIN_SILENCE_S is the hard floor — we never fire sooner than this
# regardless of score, so Condition A latency remains the minimum baseline.
# Transactional (0.8s threshold) is unaffected: its threshold is already below
# VAD_SILENCE_THRESHOLD_S so scoring never even reaches this override path.
HIGH_CONFIDENCE_THRESHOLD: float = 0.85
EARLY_FIRE_MIN_SILENCE_S:  float = VAD_SILENCE_THRESHOLD_S  # 1.0s floor

# ── Early classification (Phase 2 logging) ────────────────────────────────────
# How many speech chunks between early-classification checkpoints.
# At CHUNK_MS=300ms, every 2 chunks ≈ 600ms. Adjust to taste.
EARLY_CLASSIFY_EVERY_N_CHUNKS: int = 2

# How many consecutive checkpoints must agree before we call the label "stable".
EARLY_CLASSIFY_STABILITY_N: int = 2

# ── Opening prompt ─────────────────────────────────────────────────────────────
# Seeded into conversation history on connect. Displayed as static text in the UI.
# No TTS is triggered on connection — participant reads it and clicks the mic.
OPENING_PROMPT = (
    "Hi, I'm here to help you think through a decision. "
    "What's a decision you're currently weighing — it could be about a course, "
    "a project direction, or anything you've been going back and forth on?"
)


## Groq client singleton
# Assigned in load_models() after key validation; shared across all connections.
groq_client: AsyncGroq | None = None


## Model container 
class Models:
    vad_model = None  # Silero VAD torch module
    whisper   = None  # faster-whisper WhisperModel (local backend only)


## Per-connection state
class ConnectionState:
    """
    Holds all mutable state for a single WebSocket connection.
    One instance is created per client at the start of websocket_endpoint()
    and passed explicitly to every handler function.
    """

    def __init__(self):
        # Lifetime flag — set False when the WebSocket closes so background
        # tasks (ping loop, fire_turn pipeline) can exit cleanly.
        self.connected: bool = True

        # Audio buffers for the current turn
        self.pcm_chunks:  list[np.ndarray] = []  # decoded PCM for current turn
        self.raw_chunks:  list[bytes]      = []  # raw WebM bytes mirroring pcm_chunks

        # Rolling pre-speech buffer — holds the last PRE_SPEECH_PAD silent chunks
        # so that words spoken just before VAD fires are not clipped
        self.pre_pcm_buf: list[np.ndarray] = []
        self.pre_raw_buf: list[bytes]      = []

        # Full LLM conversation history for the session (user + assistant turns)
        self.conversation_history: list[dict] = []

        # Growing buffer of ALL raw WebM bytes received this session.
        # Chrome/Windows omits the EBML init segment so we must always send
        # the full buffer to Groq — slicing produces invalid WebM.
        self.webm_session_buf: bytes = b""

        # Transcript from the previous completed turn. Used to strip already-seen
        # text from the full-buffer transcription so partial/final results only
        # contain the current turn's words.
        self.prev_transcript: str = ""

        # Timing and flow-control
        self.silence_start: float | None = None  # monotonic time when silence began
        self.processing:    bool         = False  # True while transcription/LLM runs
        self.warmup_skip:   int          = 0      # chunks to discard after TTS

        # ── Active condition for this connection (A / B / C) ─────────────────
        # Stored per-connection so that switching condition on one client does
        # not affect other concurrent sessions or mid-turn pipeline reads.
        self.condition: str = CONDITION

        # ── Ensemble scoring state (Conditions B and C) ───────────────────────
        self.task_label:               str               = "unknown"
        self.early_classify_times:     list[tuple]       = []
        self.current_silence_threshold: float            = VAD_SILENCE_THRESHOLD_S

        # ── Turn log — one entry per completed turn ───────────────────────────
        self.turn_log: list[dict] = []

        # Tracks all in-flight early_classify_checkpoint tasks so they can be
        # cancelled on reset_turn instead of running detached and corrupting state.
        self._early_classify_tasks: list[asyncio.Task] = []

        # Timestamp when the first voiced chunk of the current turn arrived
        self.turn_start_time: float | None = None

        # Last ensemble scores — forwarded to frontend for live display
        self.last_eos_score:      float = 0.0
        self.last_llm_score:      float = 0.0
        self.last_combined_score: float = 0.0

        self._cached_partial: str = ""
        self._cached_partial_t: float = 0.0

        # Speculative transcription (Condition A): a background asyncio.Task
        # started at SPECULATIVE_TRANSCRIBE_AT_S into silence. fire_turn() uses
        # its result instead of starting a fresh Groq call, saving ~1.3s.
        self._speculative_task:       "asyncio.Task | None" = None
        self._speculative_transcript: str                   = ""

        # Non-blocking ensemble task (Conditions B/C): runs should_fire_ensemble
        # as a background task so the WS loop is never blocked waiting for
        # classify + EOS + LLM scorer calls (300-600ms total for Condition C).
        self._ensemble_task:          "asyncio.Task | None" = None
        self._ensemble_partial:       str                   = ""
        self._ensemble_delta_t:       float                 = 0.0

        # Incremental VAD decode: tracks the PCM length after the last full
        # decode so we can slice only the newly-appended tail for VAD scoring
        # rather than re-decoding the entire growing session buffer each chunk.
        # Reset each turn because speech chunk indices restart.
        self._last_decoded_pcm_len: int = 0

        # ── Per-turn latency tracking (Phase 3) ───────────────────────────────
        self.latency: dict[str, list[float]] = {
            "vad":            [],
            "transcription":  [],
            "eos_scorer":     [],
            "task_classifier":[],
            "llm_scorer":     [],
            "ensemble":       [],
            "llm_response":   [],
        }

    def reset_turn(self, warmup: int = 0) -> None:
        """
        Clears per-turn audio buffers and resets flow-control flags.
        Called after a turn fires or after TTS finishes.
        `warmup` discards the next N chunks to let echo cancellation settle.
        webm_session_buf and turn_log are intentionally preserved for the full session.
        """
        self.pcm_chunks    = []
        self.raw_chunks    = []
        self.pre_pcm_buf   = []
        self.pre_raw_buf   = []
        self.silence_start = None
        # NOTE: self.processing is intentionally NOT reset here.
        # fire_turn() is the sole owner of the processing lock; it sets
        # processing=True before calling reset_turn() and releases it in its
        # own finally block. Clearing it here would open a window where a new
        # audio chunk slips through the guard mid-pipeline.
        self.warmup_skip   = warmup

        # Cancel all in-flight early_classify_checkpoint tasks so they cannot
        # write stale labels or send spurious ws messages after the turn resets.
        for t in self._early_classify_tasks:
            if not t.done():
                t.cancel()
        self._early_classify_tasks = []

        # Reset per-turn ensemble state
        self.task_label                = "unknown"
        self.early_classify_times      = []
        self.current_silence_threshold = VAD_SILENCE_THRESHOLD_S
        self.turn_start_time           = None
        # Reset last scores
        self.last_eos_score      = 0.0
        self.last_llm_score      = 0.0
        self.last_combined_score = 0.0

        self._cached_partial = ""
        self._cached_partial_t = 0.0

        # Cancel any in-flight speculative transcription task and clear result.
        if self._speculative_task is not None and not self._speculative_task.done():
            self._speculative_task.cancel()
        self._speculative_task       = None
        self._speculative_transcript = ""

        # Cancel any in-flight ensemble scoring task.
        if self._ensemble_task is not None and not self._ensemble_task.done():
            self._ensemble_task.cancel()
        self._ensemble_task    = None
        self._ensemble_partial = ""
        self._ensemble_delta_t = 0.0

        # Reset Silero RNN hidden state so residual speech probability from the
        # previous turn does not bias VAD at the start of the next turn.
        if Models.vad_model is not None:
            try:
                Models.vad_model.reset_states()
            except Exception:
                pass

        # Reset incremental decode offset so the next turn slices from a
        # fresh baseline rather than a stale position in a longer buffer.
        self._last_decoded_pcm_len = 0

        # prev_transcript is NOT reset here — it is updated after each completed
        # turn so the next transcription can strip already-seen text.
        # Reset latency accumulators
        self.latency = {
            "vad":            [],
            "transcription":  [],
            "eos_scorer":     [],
            "task_classifier":[],
            "llm_scorer":     [],
            "ensemble":       [],
            "llm_response":   [],
        }
        log.info("Turn reset — ready. (warmup_skip=%d)", warmup)


## WebM -> PCM decoder 
# WebM EBML header magic bytes. Every valid WebM stream must start with these.
# Chrome sends the EBML init segment as the first blob on a new MediaRecorder
# session. After a reconnect, webm_session_buf is reset to b"" and the first
# few chunks may be continuation clusters without a header — PyAV raises
# "Invalid data found when processing input: '<none>'" on these.
# We check for the magic prefix before every decode attempt so we never pass
# a headerless buffer to PyAV. This fix applies to all conditions (A, B, C).
_WEBM_EBML_MAGIC = b'\x1a\x45\xdf\xa3'

def decode_to_pcm(webm_bytes: bytes) -> "np.ndarray | None":
    """
    Decodes accumulated WebM bytes to mono 16 kHz float32 PCM.

    Chrome on Windows sends WebM clusters without an EBML init segment, so
    we decode the full growing session buffer each call. The caller trims
    the result to the last chunk's worth of samples for VAD scoring.
    """
    # Guard: only attempt decode if the buffer starts with a valid EBML header.
    # Continuation clusters without the EBML prefix cause PyAV to raise
    # "[Errno 1094995529] Invalid data found when processing input: '<none>'".
    # This happens after every reconnect until the browser sends a fresh
    # MediaRecorder session that includes the init segment.
    if not webm_bytes or not webm_bytes.startswith(_WEBM_EBML_MAGIC):
        return None
    try:
        buf = io.BytesIO(webm_bytes)
        with av.open(buf) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                return None
            stream    = streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            frames = []
            for packet in container.demux(stream):
                try:
                    for frame in packet.decode():
                        for rf in resampler.resample(frame):
                            arr = rf.to_ndarray()
                            frames.append(arr[0] if arr.ndim > 1 else arr)
                except Exception:
                    continue
            # Flush resampler
            for rf in resampler.resample(None):
                arr = rf.to_ndarray()
                frames.append(arr[0] if arr.ndim > 1 else arr)
            if not frames:
                return None
            return np.concatenate(frames).astype(np.float32)
    except Exception as exc:
        log.debug("decode_to_pcm error: %s", exc)
        return None


## Latency helper
def _ms(t0: float) -> float:
    """Returns elapsed milliseconds since t0 (monotonic)."""
    return round((time.monotonic() - t0) * 1000, 2)


## VAD scoring 
def compute_vad_prob(pcm: np.ndarray, state: "ConnectionState | None" = None) -> float:
    """
    Splits PCM into 512-sample windows and runs Silero on each.
    Returns the max speech probability across all windows.
    Input shorter than one window is zero-padded.
    If state is provided, records the call latency in state.latency["vad"].
    """
    t0 = time.monotonic()
    if len(pcm) < VAD_WINDOW:
        pcm = np.pad(pcm, (0, VAD_WINDOW - len(pcm)))

    probs = []
    for start in range(0, len(pcm) - VAD_WINDOW + 1, VAD_WINDOW):
        window = torch.from_numpy(pcm[start : start + VAD_WINDOW]).unsqueeze(0)
        with torch.no_grad():
            out = Models.vad_model(window, SAMPLE_RATE)
        probs.append(float(out[0]) if isinstance(out, (tuple, list)) else float(out))

    # Use mean rather than max across windows.
    # max() returns speech=True if any single 512-sample window is high,
    # which makes VAD overly sticky after long speech: Silero's saturated
    # hidden state keeps every window near 1.0 for several chunks after you
    # stop talking. mean() requires the majority of the chunk to be speech,
    # giving cleaner silence detection at turn boundaries.
    result = (sum(probs) / len(probs)) if probs else 0.0
    if state is not None:
        state.latency["vad"].append(_ms(t0))
    return result


## Local transcription
def _transcribe_local(pcm_frames: list[np.ndarray]) -> str:
    """Concatenates PCM frames and transcribes with faster-whisper on CPU."""
    pcm = np.concatenate(pcm_frames).astype(np.float32)
    t0  = time.time()

    segs, _ = Models.whisper.transcribe(
        pcm,
        language="en",
        beam_size=3,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
        condition_on_previous_text=False,
        temperature=0.0,
    )

    result = " ".join(s.text.strip() for s in segs).strip()
    log.info("Local transcription: %.2fs → %s", time.time() - t0, result[:80])
    return result



## Speculative transcription (Condition A)
async def _run_speculative_transcription(state: "ConnectionState") -> None:
    """
    Launched as a background Task when silence reaches SPECULATIVE_TRANSCRIBE_AT_S.
    Calls Groq Whisper on the current audio buffer and stores the result in
    state._speculative_transcript so fire_turn() can use it immediately instead
    of starting a fresh call — effectively overlapping the silence wait with the
    ~1.3s Groq round-trip.

    Cancelled automatically by reset_turn() if speech resumes before the turn fires.
    """
    try:
        result = await transcribe_utterance(
            list(state.pcm_chunks),
            list(state.raw_chunks),
            state,
        )
        state._speculative_transcript = result or ""
        log.info("Speculative transcription complete: %s", state._speculative_transcript[:80])
    except asyncio.CancelledError:
        log.debug("Speculative transcription task cancelled.")
    except Exception as exc:
        log.warning("Speculative transcription error: %s", exc)
        state._speculative_transcript = ""


## Fuzzy prefix stripper
def _strip_prefix_fuzzy(full_text: str, prev_text: str) -> str:
    """
    Strips prev_text from the start of full_text using a sliding word-window
    rather than exact startswith(). Whisper is non-deterministic across calls so
    the same audio may transcribe slightly differently, causing a strict prefix
    match to fail and return the full session transcript as if it were the current
    turn (hallucinated words from earlier turns bleed into the result).
    """
    if full_text.startswith(prev_text):
        return full_text[len(prev_text):].strip()

    prev_words = prev_text.lower().split()
    full_words = full_text.split()
    if not prev_words:
        return full_text

    window_size = min(5, len(prev_words))
    tail = prev_words[-window_size:]

    best_end = -1
    for i in range(len(full_words) - window_size + 1):
        candidate = [w.lower().strip(".,!?;:\"\'") for w in full_words[i:i + window_size]]
        tail_clean = [w.strip(".,!?;:\"\'") for w in tail]
        matches = sum(a == b for a, b in zip(candidate, tail_clean))
        if matches >= max(1, window_size - 1):
            best_end = i + window_size
            break

    if best_end > 0:
        result = " ".join(full_words[best_end:]).strip()
        log.debug(
            "Fuzzy prefix strip: found prev tail at word %d — stripped %d words, kept %d.",
            best_end - window_size, best_end, len(full_words) - best_end,
        )
        return result

    log.warning(
        "Fuzzy prefix strip: no overlap found — returning full_text. prev_tail=%s  full_start=%s",
        " ".join(tail), " ".join(full_words[:8]),
    )
    return full_text


## Transcription dispatcher 
async def transcribe_utterance(
    pcm_frames: list[np.ndarray],
    raw_chunks: list[bytes],
    state: "ConnectionState | None" = None,
    final: bool = False,
) -> str:
    """
    Routes the utterance to Groq Whisper for transcription.

    Always sends the full webm_session_buf — Chrome/Windows clusters are not
    self-contained so offset slices produce invalid WebM. To isolate the current
    turn's text, we strip any prefix that matches state.prev_transcript (the
    cumulative transcript of all previous turns in this session).

    Falls back gracefully if Groq fails and local Whisper is not loaded.
    Returns an empty string if no frames or no audio bytes are available.
    If state is provided, records latency in state.latency["transcription"].
    """
    if not pcm_frames:
        return ""

    t0 = time.monotonic()

    if TRANSCRIPTION_BACKEND == "groq":
        try:
            webm_bytes = (
                state.webm_session_buf
                if state is not None and state.webm_session_buf
                else b"".join(raw_chunks)
            )

            if not webm_bytes:
                log.warning("Empty webm buffer — skipping transcription.")
                return ""

            transcription = await groq_client.audio.transcriptions.create(
                file=("audio.webm", webm_bytes, "audio/webm"),
                model=GROQ_WHISPER_MODEL,
                language="en",
                prompt=(
                    "The speaker may have a non-native English accent. "
                    "Transcribe accurately including all words."
                ),
            )
            full_text = transcription.text.strip()

            # Strip the previous turns' text to isolate the current turn.
            # Groq re-transcribes the full session buffer each call, so the
            # result always starts with everything spoken so far.
            if state is not None and state.prev_transcript:
                result = _strip_prefix_fuzzy(full_text, state.prev_transcript.strip())
            else:
                result = full_text

            if state is not None:
                state.latency["transcription"].append(_ms(t0))
            log.info("Groq transcription: %.2fs → %s", (time.monotonic() - t0), result[:80])
            return result

        except Exception as exc:
            log.warning("Groq transcription failed (%s) — falling back.", exc)

    # Local fallback — also primary path when TRANSCRIPTION_BACKEND="local"
    if Models.whisper is None:
        log.warning("Local Whisper not loaded — returning empty.")
        return ""

    result = await asyncio.to_thread(_transcribe_local, pcm_frames)
    if state is not None:
        state.latency["transcription"].append(_ms(t0))
    return result

## WebSocket send helper
async def ws_send(ws: WebSocket, msg: dict) -> None:
    """Serialises msg to JSON and sends as a text frame. Exceptions are suppressed."""
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


## Ensemble scoring functions (Conditions B and C)

async def score_eos(transcript: str, state: "ConnectionState | None" = None) -> float:
    """
    Condition B + C: Estimates syntactic turn-completion probability using the
    Groq LLM as a lightweight EOS scorer.

    Asks the model to rate on a 0-1 scale whether the transcript represents a
    complete thought. Returns the parsed float, or 0.5 on parse failure.
    If state is provided, records latency in state.latency["eos_scorer"].
    """
    if not transcript.strip():
        return 0.0
    try:
        t0 = time.monotonic()
        response = await groq_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a turn-taking classifier. "
                        "Given a speech transcript, output ONLY a single float between 0.0 and 1.0 "
                        "representing the probability that the speaker has completed their turn. "
                        "1.0 = syntactically and pragmatically complete. "
                        "0.0 = clearly mid-sentence or mid-thought. "
                        "Output only the number, nothing else."
                    ),
                },
                {"role": "user", "content": f"Transcript: {transcript}"},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw)
        score = max(0.0, min(1.0, score))
        if state is not None:
            state.latency["eos_scorer"].append(_ms(t0))
        log.info("EOS score: %.3f for: %s (%.0fms)", score, transcript[:60], _ms(t0))
        return score
    except Exception as exc:
        log.warning("EOS scorer error: %s — returning 0.5", exc)
        return 0.5


async def classify_task(
    transcript: str,
    conversation_history: list[dict],
    state: "ConnectionState | None" = None,
) -> str:
    """
    Condition C: Classifies the utterance into one of three intent types:
      - transactional  : closed-ended, single-exchange (questions, commands)
      - informational  : factual statements or explanations
      - reflective     : open-ended deliberation, thinking out loud

    Uses conversation history for pragmatic context.
    Returns one of the three labels, or "unknown" on failure.
    If state is provided, records latency in state.latency["task_classifier"].
    """
    if not transcript.strip():
        return "unknown"
    try:
        t0 = time.monotonic()
        history_snippet = "\n".join(
            f"{m['role'].upper()}: {m['content'][:100]}"
            for m in conversation_history[-4:]
        )
        response = await groq_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a turn-taking classifier for a voice AI system. "
                        "Classify the user's utterance into exactly one of these three types:\n"
                        "  transactional — closed question or command expecting a direct answer\n"
                        "  informational — factual statement or explanation\n"
                        "  reflective    — thinking out loud, deliberating, open-ended\n\n"
                        "Consider the conversation context. "
                        "Output ONLY one word: transactional, informational, or reflective."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{history_snippet}\n\nUtterance: {transcript}",
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        label = response.choices[0].message.content.strip().lower()
        if label not in ("transactional", "informational", "reflective"):
            label = "unknown"
        if state is not None:
            state.latency["task_classifier"].append(_ms(t0))
        log.info("Task label: %s for: %s (%.0fms)", label, transcript[:60], _ms(t0))
        return label
    except Exception as exc:
        log.warning("Task classifier error: %s — returning unknown", exc)
        return "unknown"


async def score_llm_turn(
    transcript: str,
    delta_t: float,
    task_label: str,
    conversation_history: list[dict],
    state: "ConnectionState | None" = None,
) -> float:
    """
    Condition C: Scores turn completion using the transcript, pause duration,
    task type, and conversation history.

    Returns a float in [0, 1] representing probability the turn is complete.
    If state is provided, records latency in state.latency["llm_scorer"].
    """
    if not transcript.strip():
        return 0.0
    try:
        t0 = time.monotonic()
        history_snippet = "\n".join(
            f"{m['role'].upper()}: {m['content'][:100]}"
            for m in conversation_history[-4:]
        )
        response = await groq_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a turn-taking scorer for a voice AI system. "
                        "Given a transcript, pause duration, task type, and conversation context, "
                        "output ONLY a single float between 0.0 and 1.0 representing the probability "
                        "that the speaker has finished their turn and the AI should respond. "
                        "1.0 = definitely done. 0.0 = definitely still speaking. "
                        "Output only the number, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{history_snippet}\n\n"
                        f"Utterance: {transcript}\n"
                        f"Pause duration: {delta_t:.2f}s\n"
                        f"Task type: {task_label}"
                    ),
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        score = float(raw)
        score = max(0.0, min(1.0, score))
        if state is not None:
            state.latency["llm_scorer"].append(_ms(t0))
        log.info("LLM turn score: %.3f (%.0fms)", score, _ms(t0))
        return score
    except Exception as exc:
        log.warning("LLM turn scorer error: %s — returning 0.5", exc)
        return 0.5


async def early_classify_checkpoint(ws: "WebSocket", state: ConnectionState) -> None:
    """
    Phase 2 — Early classification logging (Condition C only).

    Called every EARLY_CLASSIFY_EVERY_N_CHUNKS speech chunks while the user
    is still speaking. Runs the task classifier on the partial transcript
    accumulated so far and records:
      - elapsed time since turn start (ms)
      - the label at that checkpoint
      - whether the label has been stable for EARLY_CLASSIFY_STABILITY_N checks

    Sends a "task_label" message to the frontend whenever the label changes
    so the badge updates live during speech, not only at silence time.
    """
    if state.condition != "C":
        return
    if not state.pcm_chunks or state.turn_start_time is None:
        return

    elapsed_s = time.monotonic() - state.turn_start_time

    try:
        partial = await transcribe_utterance(
            list(state.pcm_chunks),
            list(state.raw_chunks),
            state,
        )
    except Exception as exc:
        log.warning("Early classify transcription error: %s", exc)
        return

    if not partial:
        return

    # Guard: if the turn was reset while we were awaiting transcription/classify,
    # discard this result rather than writing a stale label into fresh state.
    if state.turn_start_time is None:
        log.debug("Early classify: turn reset while in flight — discarding result.")
        return

    label = await classify_task(partial, state.conversation_history, state)

    if state.turn_start_time is None:
        log.debug("Early classify: turn reset while classifying — discarding result.")
        return

    # Stability check: count how many of the last N checkpoints share this label
    recent_labels = [entry["label"] for entry in state.early_classify_times[-(EARLY_CLASSIFY_STABILITY_N - 1):]]
    recent_labels.append(label)
    stable = (
        len(recent_labels) >= EARLY_CLASSIFY_STABILITY_N
        and len(set(recent_labels)) == 1
    )

    entry = {
        "elapsed_ms":   round(elapsed_s * 1000),
        "label":        label,
        "partial_text": partial[:80],
        "stable":       stable,
        "chunk_count":  len(state.pcm_chunks),
    }
    state.early_classify_times.append(entry)

    log.info(
        "Early classify checkpoint — elapsed=%.0fms label=%s stable=%s transcript=%s",
        elapsed_s * 1000, label, stable, partial[:60],
    )

    # Always update task_label and push to frontend so the badge shows
    # the current classification live during speech, not only at silence time.
    state.task_label = label
    await ws_send(ws, {
        "type":       "task_label",
        "label":      label,
        "stable":     stable,
        "elapsed_ms": round(elapsed_s * 1000),
        "dyn_threshold_s": DYNAMIC_THRESHOLDS.get(label, VAD_SILENCE_THRESHOLD_S),
    })

    if stable:
        state.current_silence_threshold = DYNAMIC_THRESHOLDS.get(label, VAD_SILENCE_THRESHOLD_S)
        log.info(
            "Early classify stable — label=%s dynamic_threshold=%.1fs",
            label, state.current_silence_threshold,
        )


async def should_fire_ensemble(
    transcript: str,
    delta_t: float,
    state: ConnectionState,
) -> bool:
    """
    Condition B + C: Decides whether to fire the turn based on the active condition.

    Condition B: EOS score alone must exceed ENSEMBLE_THRESHOLD.
    Condition C: Runs task classifier + parallel EOS + LLM scorer.
                 Combined weighted score must exceed ENSEMBLE_THRESHOLD.
                 Also enforces the dynamic silence threshold for the task type.
    """
    if state.condition == "B":
        eos_score = await score_eos(transcript, state)
        decision  = eos_score >= ENSEMBLE_THRESHOLD
        # Store scores so the WS loop can snapshot them for the frontend scoring panel.
        state.last_eos_score      = eos_score
        state.last_llm_score      = 0.0   # LLM scorer not used in B
        state.last_combined_score = eos_score  # combined == EOS for B
        log.info("Condition B — EOS=%.3f threshold=%.2f fire=%s", eos_score, ENSEMBLE_THRESHOLD, decision)
        return decision

    # Condition C
    label = await classify_task(transcript, state.conversation_history, state)
    state.task_label                = label
    state.current_silence_threshold = DYNAMIC_THRESHOLDS.get(label, VAD_SILENCE_THRESHOLD_S)

    # ── Two-tier firing decision ───────────────────────────────────────────────
    # Always score — we need the combined value to decide which tier applies.
    # The threshold check is moved AFTER scoring so high-confidence utterances
    # can fire before the full dynamic window expires.
    #
    # Exception: if we haven't even hit the early-fire floor yet, skip scoring
    # entirely — there isn't enough silence to warrant a response regardless.
    if delta_t < EARLY_FIRE_MIN_SILENCE_S:
        log.info(
            "Condition C — δ_t=%.2fs < early-fire floor=%.2fs — holding (no score)",
            delta_t, EARLY_FIRE_MIN_SILENCE_S,
        )
        state.last_eos_score      = 0.0
        state.last_llm_score      = 0.0
        state.last_combined_score = 0.0
        return False

    t0_ensemble = time.monotonic()
    eos_score, llm_score = await asyncio.gather(
        score_eos(transcript, state),
        score_llm_turn(transcript, delta_t, label, state.conversation_history, state),
    )

    combined = EOS_WEIGHT * eos_score + LLM_WEIGHT * llm_score
    state.latency["ensemble"].append(_ms(t0_ensemble))

    state.last_eos_score      = eos_score
    state.last_llm_score      = llm_score
    state.last_combined_score = combined

    # Tier 1 — high confidence: fire immediately, regardless of dynamic threshold.
    # Applies to all task types: transactional fires at its own 0.8s floor anyway,
    # so the meaningful savings are on informational (saves up to 0.8s) and
    # reflective (saves up to 2.0s).
    if combined >= HIGH_CONFIDENCE_THRESHOLD and delta_t >= EARLY_FIRE_MIN_SILENCE_S:
        decision = True
        log.info(
            "Condition C — label=%s EOS=%.3f LLM=%.3f combined=%.3f ≥ high-conf=%.2f "
            "— EARLY FIRE at δ_t=%.2fs (dynamic threshold would have been %.2fs)",
            label, eos_score, llm_score, combined, HIGH_CONFIDENCE_THRESHOLD,
            delta_t, state.current_silence_threshold,
        )
        return decision

    # Tier 2 — normal confidence: require the dynamic threshold to be met first.
    if delta_t < state.current_silence_threshold:
        log.info(
            "Condition C — label=%s EOS=%.3f LLM=%.3f combined=%.3f "
            "— below high-conf threshold, δ_t=%.2fs < dynamic=%.2fs — holding",
            label, eos_score, llm_score, combined,
            delta_t, state.current_silence_threshold,
        )
        return False

    decision = combined >= ENSEMBLE_THRESHOLD
    log.info(
        "Condition C — label=%s EOS=%.3f LLM=%.3f combined=%.3f threshold=%.2f fire=%s "
        "(dynamic threshold met at δ_t=%.2fs)",
        label, eos_score, llm_score, combined, ENSEMBLE_THRESHOLD, decision, delta_t,
    )
    return decision


## LLM response generation 
async def generate_and_speak(ws: WebSocket, state: ConnectionState, transcript: str) -> None:
    """
    Appends the transcript to conversation history, calls the Groq LLM, and
    sends the reply as a "response" message.

    Does NOT manage state.processing or call reset_turn().
    fire_turn() is the single owner of the processing lock and reset lifecycle;
    this function is awaited inline by fire_turn(), not as a background task.
    """
    await ws_send(ws, {"type": "status", "text": "[PROCESSING] Generating response ..."})

    # ── Task-label-aware response instruction (Condition C only) ──────────────
    # Maps the classified utterance type to an explicit behavioural instruction
    # appended to the system prompt. This prevents the LLM from defaulting to
    # its advisor persona (ask a follow-up) when the user just wanted a direct
    # answer. For Conditions A/B the label stays "unknown" and no instruction
    # is injected, preserving the original behaviour.
    TASK_RESPONSE_INSTRUCTIONS: dict[str, str] = {
        "transactional": (
            "The user asked a direct, closed question. "
            "Answer it in one sentence only. "
            "Do NOT ask a follow-up question or connect it back to their decision."
        ),
        "informational": (
            "The user made a factual statement or gave an explanation. "
            "Acknowledge it briefly, then ask one relevant probing question if appropriate."
        ),
        "reflective": (
            "The user is thinking out loud or deliberating. "
            "Ask a single open-ended probing question to help them explore their reasoning further. "
            "Do not rush to give an answer."
        ),
    }
    task_instruction = TASK_RESPONSE_INSTRUCTIONS.get(state.task_label, "")

    base_system_prompt = (
        "You are a thoughtful academic advisor helping a graduate student "
        "think through a decision they are facing. "
        "Your role is to help them deliberate, not to give them answers. "
        "Ask probing questions that draw out their reasoning. "
        "When they are thinking out loud, wait for them to finish before responding — "
        "do not rush them. "
        "When they ask a direct factual question, answer it briefly and clearly. "
        "Keep all responses concise and natural-sounding since they will be spoken aloud. "
        "Never give lists — speak in natural sentences as you would in conversation. "
        "Aim for 1-2 sentences unless a longer response is clearly needed."
    )
    system_prompt = (
        base_system_prompt + "\n\n" + task_instruction
        if task_instruction
        else base_system_prompt
    )

    log.info(
        "generate_and_speak: task_label=%s instruction_injected=%s",
        state.task_label, bool(task_instruction),
    )

    state.conversation_history.append({"role": "user", "content": transcript})
    payload = [
        {
            "role": "system",
            "content": system_prompt,
        },
        *state.conversation_history,
    ]

    try:
        t0 = time.monotonic()
        response = await groq_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=payload,
            max_tokens=512,
            temperature=0.7,
        )
        reply = response.choices[0].message.content.strip()
        elapsed_llm = _ms(t0)
        state.latency["llm_response"].append(elapsed_llm)
        state.conversation_history.append({"role": "assistant", "content": reply})
        log.info("LLM response: %s (%.0fms)", reply[:100], elapsed_llm)
        await ws_send(ws, {
            "type":            "response",
            "text":            reply,
            "response_ms":     round(elapsed_llm),
        })

    except Exception as exc:
        log.error("Groq LLM error: %s", exc)
        await ws_send(ws, {"type": "status", "text": f"[ERROR] Groq error: {exc}"})


## Turn firing 
async def fire_turn(
    ws: WebSocket,
    state: ConnectionState,
    prefetched_transcript: str | None = None,
    mid_speech: bool = False,
) -> None:
    """
    Single-owner pipeline: lock → transcribe → log → LLM → release.

    processing=True is set immediately so the WS loop drops all audio during
    the transcription await (previously reset_turn cleared it with no lock held,
    causing a race where echo or next-turn audio accumulated incorrectly).

    generate_and_speak() is awaited inline — not a background task — so there
    is no race between transcription and LLM completion. The finally block is
    the single exit point for the processing lock.
    """
    # 1. Lock immediately.
    state.processing = True

    # Reset Silero hidden state NOW, before any new audio arrives.
    # Doing it here (rather than in reset_turn's finally) means the very first
    # silent chunk after this turn is scored with a clean RNN state, not with
    # the momentum built up from 70+ speech chunks at prob=1.000.
    if Models.vad_model is not None:
        try:
            Models.vad_model.reset_states()
        except Exception:
            pass

    frames = list(state.pcm_chunks)
    raw    = list(state.raw_chunks)

    turn_start      = state.turn_start_time
    task_label      = state.task_label
    early_times     = list(state.early_classify_times)
    dyn_threshold   = state.current_silence_threshold
    silence_elapsed = (
        (time.monotonic() - state.silence_start)
        if state.silence_start is not None else None
    )

    # Clear audio buffers; processing stays True.
    state.reset_turn()

    try:
        await ws_send(ws, {"type": "status", "text": "[TRANSCRIBING] ..."})

        # 2. Transcribe.
        if prefetched_transcript is not None:
            final = prefetched_transcript
        else:
            final = await transcribe_utterance(frames, raw, state, final=True)

        # Abort silently if the client disconnected while we were transcribing.
        # This prevents stale LLM calls and prev_transcript updates that would
        # corrupt the next session's prefix stripping.
        if not state.connected:
            log.info("fire_turn: client disconnected during transcription — aborting.")
            return

        if not final:
            log.info("Empty transcript after transcription — resetting.")
            await ws_send(ws, {"type": "status", "text": "[READY] Click the mic and start speaking"})
            return

        # 3. Update cumulative prev_transcript.
        if not mid_speech:
            if state.prev_transcript:
                state.prev_transcript = (state.prev_transcript + " " + final).strip()
            else:
                state.prev_transcript = final

        log.info("Final transcript: %s", final)
        await ws_send(ws, {"type": "transcript", "text": final})

        # 4. Log.
        first_stable = next(
            (e["elapsed_ms"] for e in early_times if e.get("stable")), None
        )

        def _lat_summary(vals: list[float]) -> dict:
            if not vals:
                return {"n": 0, "mean_ms": None, "max_ms": None}
            return {
                "n":       len(vals),
                "mean_ms": round(sum(vals) / len(vals), 1),
                "max_ms":  round(max(vals), 1),
            }

        latency_snapshot = {k: _lat_summary(v) for k, v in state.latency.items()}

        log_entry = {
            "timestamp":            time.time(),
            "condition":            state.condition,
            "transcript":           final,
            "task_label":           task_label,
            "silence_at_fire_s":    round(silence_elapsed, 3) if silence_elapsed else None,
            "dynamic_threshold":    dyn_threshold,
            "turn_duration_s":      round(time.monotonic() - turn_start, 3) if turn_start else None,
            "early_classify_times": early_times,
            "first_stable_ms":      first_stable,
            "chunk_count":          len(frames),
            "latency":              latency_snapshot,
        }
        state.turn_log.append(log_entry)
        log.info("Turn log entry: %s", json.dumps(log_entry))

        # 5. Generate and speak — awaited inline, single lock owner.
        await generate_and_speak(ws, state, final)

    finally:
        # Single exit point: release lock and apply warmup.
        state.processing = False
        state.warmup_skip = 5
        log.info("fire_turn: pipeline complete, warmup=5 applied.")


## Model loading 
def load_models() -> None:
    """Loads Silero VAD and (optionally) faster-whisper. Runs once at startup."""
    global groq_client

    if TRANSCRIPTION_BACKEND == "groq" and not GROQ_API_KEY:
        log.warning(
            "GROQ_API_KEY is not set — Groq backend will fail at runtime. "
            "Set the environment variable and restart, or switch "
            "TRANSCRIPTION_BACKEND to 'local'."
        )

    if GROQ_API_KEY:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        log.info("AsyncGroq client initialised.")

    log.info("Loading Silero VAD …")
    Models.vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    Models.vad_model.eval()

    if TRANSCRIPTION_BACKEND == "local":
        log.info("Loading faster-whisper (%s / %s) …", WHISPER_MODEL_SIZE, WHISPER_COMPUTE)
        cpu_threads = max(4, os.cpu_count() or 4)
        Models.whisper = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
            cpu_threads=cpu_threads,
            num_workers=1,
        )
        log.info("Whisper loaded — using %d CPU threads.", cpu_threads)
    else:
        log.info("Transcription backend: Groq %s — local Whisper not loaded.", GROQ_WHISPER_MODEL)

    log.info("All models ready.")


## FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Calls load_models() before the server accepts requests."""
    load_models()
    yield


app = FastAPI(title="VAD Baseline", lifespan=lifespan)


## Session log export
_last_session_log: list[dict] = []  # set by websocket_endpoint on disconnect


@app.get("/session-log")
async def get_session_log():
    """Returns the turn log from the most recently completed session as JSON."""
    from fastapi.responses import JSONResponse
    return JSONResponse(content=_last_session_log)


@app.get("/session-log/clear")
async def clear_session_log():
    """Clears the cached session log. Call before starting a new participant."""
    global _last_session_log
    _last_session_log = []
    return {"status": "cleared"}


@app.get("/condition")
async def get_condition():
    """Returns the server-wide default condition for new connections."""
    return {"condition": CONDITION}

@app.post("/condition/{val}")
async def set_condition_http(val: str):
    """
    Sets the server-wide default condition for new connections via HTTP POST.
    Already-connected sessions are not affected; use the WebSocket
    set_condition message to change condition mid-session.
    """
    global CONDITION
    val = val.upper().strip()
    if val not in ("A", "B", "C"):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={"error": f"Invalid condition: {val}"})
    CONDITION = val
    log.info("Default condition set to %s via HTTP.", CONDITION)
    return {"condition": CONDITION}


@app.get("/latency-summary")
async def get_latency_summary():
    """
    Aggregates latency data across all turns in the last session.
    Returns mean and max per component.
    """
    from fastapi.responses import JSONResponse

    if not _last_session_log:
        return JSONResponse(content={"error": "No session data available."})

    component_buckets: dict[str, list[float]] = {
        "vad": [], "transcription": [], "eos_scorer": [],
        "task_classifier": [], "llm_scorer": [], "ensemble": [], "llm_response": [],
    }

    for entry in _last_session_log:
        lat = entry.get("latency", {})
        for component, stats in lat.items():
            if stats.get("mean_ms") is not None:
                component_buckets.setdefault(component, []).append(stats["mean_ms"])

    def _agg(vals: list[float]) -> dict:
        if not vals:
            return {"mean_ms": None, "max_ms": None, "n_turns": 0}
        return {
            "mean_ms": round(sum(vals) / len(vals), 1),
            "max_ms":  round(max(vals), 1),
            "n_turns": len(vals),
        }

    summary = {
        "condition":   _last_session_log[0].get("condition", "?") if _last_session_log else "?",
        "turn_count":  len(_last_session_log),
        "components":  {k: _agg(v) for k, v in component_buckets.items()},
    }
    return JSONResponse(content=summary)


## Frontend route
FRONTEND_PATH = Path(__file__).parent / "index.html"

@app.get("/")
async def root():
    """Serves index.html from the same directory as this script."""
    if FRONTEND_PATH.exists():
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found next to server.py</h1>")


## WebSocket endpoint 
async def _ping_loop(ws: WebSocket, state: "ConnectionState") -> None:
    """
    Sends a WebSocket ping every 15 seconds to prevent the browser from
    closing an idle connection (Chrome drops quiet WebSockets after ~30s).
    Exits cleanly when state.connected is set to False.
    """
    while state.connected:
        await asyncio.sleep(15)
        if not state.connected:
            break
        try:
            await ws.send_text(json.dumps({"type": "ping"}))
        except Exception:
            break


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Entry point for each browser connection. Creates a ConnectionState instance
    for the client, then runs the receive loop until the client disconnects.
    A background ping loop keeps the connection alive through idle periods.
    """
    await ws.accept()
    log.info("Client connected.")

    state = ConnectionState()
    state.connected = True

    # Seed the opening prompt into conversation history.
    # No TTS is triggered — the prompt is shown as static text in the UI.
    state.conversation_history.append({"role": "assistant", "content": OPENING_PROMPT})

    await ws_send(ws, {"type": "status", "text": "[READY] Click the mic and start speaking"})

    ping_task = asyncio.create_task(_ping_loop(ws, state))

    try:
        while True:
            try:
                data = await ws.receive()
            except (WebSocketDisconnect, RuntimeError):
                break

            raw  = data.get("bytes")
            text = data.get("text")

            # JSON control messages 
            if text:
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "set_condition":
                    val = msg.get("condition", "").upper().strip()
                    if val in ("A", "B", "C"):
                        state.condition = val
                        log.info("Condition switched to %s by frontend.", state.condition)
                        await ws_send(ws, {"type": "condition_ack", "condition": state.condition})
                    else:
                        log.warning("Invalid condition value received: %s", val)
                    continue

                if msg.get("type") == "reset":
                    # Sent by the frontend after TTS onend fires.
                    # The Web Speech API can fire onend multiple times on long
                    # utterances (internal TTS segmentation), causing spurious
                    # resets that wipe audio mid-speech on the next turn.
                    # Guard: if warmup or processing is already active from
                    # fire_turn's own reset, ignore this duplicate reset entirely.
                    if state.warmup_skip > 0 or state.processing:
                        log.debug(
                            "Frontend reset ignored — warmup_skip=%d processing=%s",
                            state.warmup_skip, state.processing,
                        )
                    elif state.pcm_chunks:
                        # User is already speaking — do not reset mid-speech.
                        log.debug("Frontend reset ignored — speech already in progress (%d chunks).", len(state.pcm_chunks))
                    else:
                        state.reset_turn(warmup=5)
                        log.info("Frontend reset accepted — warmup=5 applied.")
                    await ws_send(ws, {"type": "status", "text": "[READY] Listening ..."})

                elif msg.get("type") == "clear_history":
                    state.conversation_history.clear()
                    state.conversation_history.append({"role": "assistant", "content": OPENING_PROMPT})
                    state.reset_turn()
                    log.info("Conversation history cleared.")
                    await ws_send(ws, {"type": "status", "text": "[READY] History cleared -- start speaking"})

                continue

            # Binary audio frames
            log.debug(
                "WS recv: bytes_len=%s  processing=%s  warmup=%s",
                len(raw) if raw else None, state.processing, state.warmup_skip,
            )

            if not raw or state.processing:
                continue

            if state.warmup_skip > 0:
                state.warmup_skip -= 1
                log.debug("Warmup skip: %d remaining.", state.warmup_skip)
                continue

            now = time.monotonic()

            # Accumulate raw chunks into the session buffer.
            # Only start accumulating once we've seen the EBML magic header —
            # chunks arriving before the header (e.g. continuation clusters from
            # a previous MediaRecorder that arrived after reconnect) are discarded.
            # This prevents headerless data from polluting the new session buffer
            # and causing PyAV decode errors on all subsequent chunks.
            if not state.webm_session_buf:
                if not raw.startswith(_WEBM_EBML_MAGIC):
                    log.debug("Discarding pre-header chunk (%d bytes) — waiting for EBML magic.", len(raw))
                    continue
                log.debug("EBML header received — starting new WebM session buffer.")
            state.webm_session_buf += raw

            full_pcm = decode_to_pcm(state.webm_session_buf)
            if full_pcm is None:
                log.debug("Chunk not decodable — skipping.")
                continue

            # Incremental VAD slice: take only the samples appended since the
            # last decode rather than always taking the last N samples of the
            # full buffer. This is O(1) regardless of session length and avoids
            # PyAV resampler flush drift that causes the tail slice to land on
            # slightly older audio after many turns (which shows as VAD=0.001
            # while the user is actively speaking).
            samples_per_chunk = int(SAMPLE_RATE * CHUNK_MS / 1000)
            prev_len = state._last_decoded_pcm_len
            state._last_decoded_pcm_len = len(full_pcm)

            if prev_len > 0 and len(full_pcm) > prev_len:
                pcm = full_pcm[prev_len:]
                if len(pcm) > samples_per_chunk:
                    pcm = pcm[-samples_per_chunk:]
            else:
                # First chunk of session or decode returned fewer samples —
                # fall back to tail slice
                pcm = full_pcm[-samples_per_chunk:] if len(full_pcm) > samples_per_chunk else full_pcm

            # Score the chunk with Silero VAD
            prob      = compute_vad_prob(pcm, state)
            is_speech = prob > SPEECH_PROB_THRESHOLD
            log.info(
                "VAD prob=%.3f  speech=%s  buffered_chunks=%d",
                prob, is_speech, len(state.pcm_chunks),
            )

            if is_speech:
                state.silence_start = None  # reset silence timer on any voiced chunk

                if not state.pcm_chunks:
                    # First voiced chunk — record turn start and prepend pre-speech buffer.
                    state.turn_start_time = now
                    state.pcm_chunks.extend(state.pre_pcm_buf)
                    state.raw_chunks.extend(state.pre_raw_buf)
                    state.pre_pcm_buf.clear()
                    state.pre_raw_buf.clear()

                    # Condition C: immediately reset the badge to unknown so the
                    # previous turn's stale label is not shown while the new turn
                    # is being classified. early_classify_checkpoint will update
                    # it once the first partial transcript is ready.
                    if state.condition == "C":
                        await ws_send(ws, {
                            "type":            "task_label",
                            "label":           "unknown",
                            "stable":          False,
                            "elapsed_ms":      0,
                            "dyn_threshold_s": VAD_SILENCE_THRESHOLD_S,
                        })

                state.pcm_chunks.append(pcm)
                state.raw_chunks.append(raw)
                count = len(state.pcm_chunks)

                await ws_send(ws, {"type": "status", "text": f"[MIC] Listening ... ({count} chunks)"})
                log.info("Speech chunk accumulated: total=%d", count)

                # ── Early classification checkpoint (Phase 2, Condition C only) ──
                # Note: `not state.processing` is intentionally omitted here.
                # processing=True belongs to the current turn's fire_turn lock;
                # by the time new speech chunks arrive it is always False.
                # The real stale-turn guard is the `turn_start_time is None`
                # check inside early_classify_checkpoint itself.
                if (
                    state.condition == "C"
                    and count % EARLY_CLASSIFY_EVERY_N_CHUNKS == 0
                ):
                    task = asyncio.create_task(early_classify_checkpoint(ws, state))
                    state._early_classify_tasks.append(task)

                if count >= MAX_SPEECH_CHUNKS:
                    log.info("Max speech chunks reached — forcing turn end.")
                    await fire_turn(ws, state, mid_speech=True)

            else:
                if not state.pcm_chunks:
                    # No speech yet — maintain the rolling pre-speech buffer.
                    state.pre_pcm_buf.append(pcm)
                    state.pre_raw_buf.append(raw)
                    if len(state.pre_pcm_buf) > PRE_SPEECH_PAD:
                        state.pre_pcm_buf.pop(0)
                        state.pre_raw_buf.pop(0)
                    continue

                # Speech already detected — accumulate silence into the utterance
                state.pcm_chunks.append(pcm)
                state.raw_chunks.append(raw)

                if state.silence_start is None:
                    state.silence_start = now
                delta_t = now - state.silence_start

                await ws_send(ws, {"type": "silence", "delta_t": round(delta_t, 2)})
                log.info("Silence δ_t=%.2fs  total_chunks=%d", delta_t, len(state.pcm_chunks))

                # ── Condition routing ──────────────────────────────────────────
                if state.condition == "A":
                    # Speculative transcription: kick off a Groq call at
                    # SPECULATIVE_TRANSCRIBE_AT_S so the result is ready by the
                    # time the silence threshold fires. Only launch once per turn.
                    if (
                        delta_t >= SPECULATIVE_TRANSCRIBE_AT_S
                        and state._speculative_task is None
                        and state.pcm_chunks
                    ):
                        log.info(
                            "Condition A — δ_t=%.2fs ≥ %.2fs — launching speculative transcription.",
                            delta_t, SPECULATIVE_TRANSCRIBE_AT_S,
                        )
                        state._speculative_task = asyncio.create_task(
                            _run_speculative_transcription(state)
                        )

                    if delta_t >= VAD_SILENCE_THRESHOLD_S:
                        log.info(
                            "Condition A — silence %.2fs ≥ %.2fs — firing turn.",
                            delta_t, VAD_SILENCE_THRESHOLD_S,
                        )
                        # Wait for speculative transcription using short poll intervals
                        # so we can detect a disconnect and abort rather than
                        # blocking the WS loop for up to 2s with wait_for().
                        # Groq can take 1.5-2s on a slow day so give it 3s total.
                        prefetched: str | None = None
                        if state._speculative_task is not None:
                            if not state._speculative_task.done():
                                log.info("Condition A — waiting for speculative transcription...")
                                deadline = time.monotonic() + 3.0
                                while not state._speculative_task.done():
                                    if not state.connected:
                                        log.info("Condition A — disconnected while waiting for speculative transcription — aborting.")
                                        state._speculative_task.cancel()
                                        break
                                    if time.monotonic() > deadline:
                                        log.warning("Condition A — speculative transcription timed out; fire_turn will re-transcribe.")
                                        break
                                    await asyncio.sleep(0.05)
                            if state._speculative_transcript:
                                prefetched = state._speculative_transcript
                                log.info("Condition A — using speculative transcript: %s", prefetched[:80])

                        if not state.connected:
                            log.info("Condition A — skipping fire_turn, client disconnected.")
                        else:
                            await fire_turn(ws, state, prefetched_transcript=prefetched)

                else:
                    # Hard cap — fire unconditionally regardless of ensemble score.
                    if delta_t >= MAX_SILENCE_S:
                        log.info(
                            "Condition %s — silence hard cap %.2fs reached — firing turn.",
                            state.condition, delta_t,
                        )
                        await fire_turn(ws, state)

                    elif delta_t >= VAD_SILENCE_THRESHOLD_S:
                        # ── Transcription (non-blocking) ──────────────────────────
                        # Never await transcription inline — it takes ~1.3s and
                        # would suspend the WS loop, starving the ensemble task of
                        # the chunks it needs to complete and be harvested.
                        # Instead: launch transcription as a background task on
                        # first silence chunk, harvest on completion.
                        if state._speculative_task is None and state.pcm_chunks:
                            state._speculative_task = asyncio.create_task(
                                _run_speculative_transcription(state)
                            )
                            log.debug("Condition %s — transcription task launched.", state.condition)

                        # Use cached partial if speculative already finished,
                        # else use whatever we have from _cached_partial.
                        if state._speculative_transcript:
                            partial = state._speculative_transcript
                            state._cached_partial     = partial
                            state._cached_partial_t   = time.monotonic()
                        elif state._cached_partial:
                            partial = state._cached_partial
                        else:
                            partial = None

                        if partial:
                            # ── Step 1: harvest completed ensemble task ────────────
                            if state._ensemble_task is not None and state._ensemble_task.done():
                                try:
                                    fire_decision = state._ensemble_task.result()
                                except Exception:
                                    fire_decision = False

                                # Snapshot everything BEFORE fire_turn wipes state
                                scored_partial  = state._ensemble_partial
                                scored_label    = state.task_label
                                scored_eos      = state.last_eos_score
                                scored_llm      = state.last_llm_score
                                scored_combined = state.last_combined_score
                                scored_dyn      = state.current_silence_threshold
                                state._ensemble_task = None

                                if scored_label != "unknown":
                                    await ws_send(ws, {
                                        "type":            "task_label",
                                        "label":           scored_label,
                                        "stable":          True,
                                        "elapsed_ms":      0,
                                        "dyn_threshold_s": scored_dyn,
                                    })

                                await ws_send(ws, {
                                    "type":             "scoring",
                                    "task_label":       scored_label,
                                    "eos_score":        round(scored_eos, 3),
                                    "llm_score":        round(scored_llm, 3),
                                    "combined_score":   round(scored_combined, 3),
                                    "threshold":        ENSEMBLE_THRESHOLD,
                                    "high_conf_threshold": HIGH_CONFIDENCE_THRESHOLD,
                                    "dyn_threshold_s":  scored_dyn,
                                    "fire":             fire_decision,
                                    # True when high-confidence override fired before the dynamic threshold
                                    "early_fire": fire_decision and (
                                        scored_combined >= HIGH_CONFIDENCE_THRESHOLD
                                        and state._ensemble_delta_t < scored_dyn
                                    ),
                                })

                                if fire_decision:
                                    log.info("Condition %s — ensemble approved — firing.", state.condition)
                                    await fire_turn(ws, state, prefetched_transcript=scored_partial)
                                    continue
                                else:
                                    log.info("Condition %s — ensemble held.", state.condition)

                            # ── Step 2: launch ensemble task if none running ────────
                            if state._ensemble_task is None:
                                state._ensemble_partial = partial
                                state._ensemble_delta_t = delta_t
                                state._ensemble_task = asyncio.create_task(
                                    should_fire_ensemble(partial, delta_t, state)
                                )
                                log.info("Condition %s — ensemble task launched (partial=%s).", state.condition, partial[:40] if partial else "")
                            else:
                                log.info("Condition %s — ensemble task still running, skipping chunk.", state.condition)
                        else:
                            log.info("Condition %s — no partial transcript yet, waiting.", state.condition)

    except Exception as exc:
        log.exception("Unhandled error in WebSocket handler: %s", exc)
    finally:
        state.connected = False
        ping_task.cancel()
        # Save this session's turn log. Use a per-connection copy so that
        # a second client disconnecting concurrently does not overwrite a
        # completed session's log before it has been downloaded.
        # _last_session_log always reflects the most recently ended session.
        global _last_session_log
        _last_session_log = list(state.turn_log)
        log.info("Session ended — %d turn log entries saved.", len(_last_session_log))
        log.info("Client disconnected.")


## Entry point
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)