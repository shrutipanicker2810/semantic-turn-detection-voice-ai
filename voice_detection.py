#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sounddevice as sd
import numpy as np
import torch
import time
from threading import Timer
import sys

# ====================== 配置 ======================
SAMPLE_RATE = 16000
BLOCK_SIZE = 512                       # ← 关键修复：Silero VAD 必须是 512 samples
CHUNK_DURATION = BLOCK_SIZE / SAMPLE_RATE   # ≈ 0.032 秒（32ms）

VAD_THRESHOLD = 0.5                    # 说话概率阈值
SILENCE_TIMEOUT = 1.0                  # 静音 1 秒后认为一句话结束

print("🚀 正在加载 Silero VAD 模型（已修正）...")
torch.set_num_threads(1)

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,                # 第一次可改为 True
    onnx=True,                         # 强烈推荐！速度更快、更稳定
    verbose=False
)

# ====================== 全局状态 ======================
utterance_buffer = []      # 当前一句话的音频块
silence_timer = None
is_speaking = False

def reset_utterance():
    """清空当前句子的音频缓存"""
    global utterance_buffer
    utterance_buffer = []

def on_silence_timeout():
    """1 秒静音超时 → 认为一句话完整结束"""
    global silence_timer
    if len(utterance_buffer) > 0:
        duration_sec = len(utterance_buffer) * CHUNK_DURATION
        print(f"\na sentence detected  duration: {duration_sec:.1f} sec | audio chunk: {len(utterance_buffer)}")
        # ==================== 后面改进时在这里加 ASR 等 ====================
        # text = asr.transcribe(utterance_buffer)
        # print("识别结果：", text)
        reset_utterance()
    silence_timer = None

def audio_callback(indata, frames, time_info, status):
    """每 32ms 调用一次"""
    global is_speaking, silence_timer

    if status:
        print(status)

    audio_chunk = indata[:, 0].astype(np.float32)

    # ====================== Silero VAD ======================
    speech_prob = model(torch.from_numpy(audio_chunk), SAMPLE_RATE).item()

    if speech_prob > VAD_THRESHOLD:          # 说话中
        utterance_buffer.append(audio_chunk.copy())
        is_speaking = True

        if silence_timer is not None:
            silence_timer.cancel()
            silence_timer = None

    else:                                    # 静音
        if is_speaking and silence_timer is None:
            silence_timer = Timer(SILENCE_TIMEOUT, on_silence_timeout)
            silence_timer.start()
            is_speaking = False

# ====================== 主程序 ======================


try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    ):
        print("mic is on, listening")
        while True:
            time.sleep(0.2)

except KeyboardInterrupt:
    print("\n\nsystem log out")
    if silence_timer:
        silence_timer.cancel()
    sys.exit(0)

except Exception as e:
    print(f"error: {e}")