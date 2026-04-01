# Semantic Turn Detection for Voice AI Systems
A real-time voice AI system that reimagines how conversational agents decide when to respond. Most deployed voice assistants rely on a fixed silence threshold; they wait for the mic to go quiet and fire after a set number of seconds, regardless of whether the speaker has actually finished their thought. This project builds a richer alternative: a pipeline that listens to what you say, not just when you stop saying it.

The system classifies each utterance into one of three intent types - transactional (a direct question expecting a short answer), informational (context-sharing or explanation), or reflective (thinking out loud, deliberating) and uses that classification to dynamically adjust how long it waits before responding. Short, closed questions get a tight window. Reflective turns, where pauses are a natural part of the thinking process, get more space. On top of that, two scoring signals run in parallel: a syntactic end-of-sentence scorer that checks whether the utterance forms a grammatically complete thought, and a semantic LLM turn scorer that considers the full conversational context. Their weighted combination drives the final fire decision.

The pipeline is built on open-source tools - Silero VAD for speech detection, Groq Whisper for transcription, and Groq LLaMA for scoring and response generation and runs entirely on CPU with no GPU or paid infrastructure required beyond the Groq free tier.

To evaluate whether this approach actually improves the conversational experience, we are running a within-subjects user study comparing three variants of the system: a silence-only VAD baseline, a VAD + syntactic EOS scorer, and the full semantic ensemble. Participants hold a deliberative conversation with each variant and rate their experience. The study tests whether semantic turn detection produces more natural, less interrupted interactions and whether the effect differs by utterance type.

## Conditions

| | Condition A | Condition B | Condition C |
|---|---|---|---|
| **Name** | VAD Baseline | VAD + Syntactic Scorer | Full Ensemble |
| **Turn trigger** | Fixed 1.0s silence threshold | EOS score ≥ 0.65 | Weighted ensemble ≥ 0.65 |
| **Scoring** | None | Syntactic EOS only | EOS (0.2) + LLM turn scorer (0.8) |
| **Intent classification** | No | No | Yes - transactional / informational / reflective |
| **Dynamic threshold** | No | No | Yes - 0.8s / 1.8s / 3.0s by intent |
| **Speculative transcription** | At 0.5s silence | - | - |
| **Onset transcription** | - | At δ_t = 0 | At δ_t = 0 |
| **Hard cap** | - | 4.0s | 4.0s |

---

## System Architecture

```
Browser (Chrome)
  │  WebM/Opus audio chunks (300ms)
  
FastAPI WebSocket server
  ├── PyAV - WebM -> mono 16kHz PCM
  ├── Silero VAD - speech probability per chunk
  │
  ├── Condition A 
  │     Speculative transcription at 0.5s silence (Groq Whisper)
  │     Fire at 1.0s silence
  │
  ├── Condition B 
  │     Onset transcription at δ_t=0 (Groq Whisper, 0.5s sleep)
  │     EOS scorer (Groq LLaMA) every 0.5s silence bucket
  │     Fire when EOS ≥ 0.65 or hard cap at 4.0s
  │
  └── Condition C 
        Early intent classification every 2 speech chunks
        Onset transcription at δ_t=0
        Parallel EOS + LLM turn scorer (Groq LLaMA)
        Combined score = 0.2·EOS + 0.8·LLM
        Tier 1: combined ≥ 0.85 -> early fire after 1.0s floor
        Tier 2: combined ≥ 0.65 after dynamic threshold
        Hard cap at 4.0s
```

**Models & APIs**
- VAD: [Silero VAD](https://github.com/snakers4/silero-vad) (JIT, CPU)
- Transcription: Groq `whisper-large-v3-turbo`
- Scoring & classification: Groq `llama-3.3-70b-versatile`
- TTS: Web Speech API (browser-side)

---

## Study Design

- **Design:** Within-subjects, fully counterbalanced (6 orderings)
- **N:** ~18 participants
- **Task:** Open-ended decision deliberation with an AI thinking partner
- **Measures:** Perceived naturalness, trust, cognitive load (NASA-TLX), turn detection accuracy

---

## Repo Structure

```
├── server.py           # FastAPI + WebSocket backend
├── index.html          # Single-file frontend
├── requirements.txt    # Python dependencies
├── railway.toml        # Railway deployment config
├── runtime.txt         # Python version pin
└── silero_vad.jit      # Bundled Silero VAD model weights
```

---

## Running Locally

**Requirements:** Python 3.11, a Groq API key (free tier works)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY=your_key_here   # Windows: $env:GROQ_API_KEY=your_key_here

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8000

# Open in Chrome
http://localhost:8000
```

> Chrome is required - other browsers have inconsistent WebM/Opus MediaRecorder support.

---

## License

MIT - model weights are MIT-licensed by the Silero team.
