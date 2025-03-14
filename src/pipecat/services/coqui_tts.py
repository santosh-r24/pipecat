#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import numpy as np
import torch
import torchaudio
from loguru import logger
from typing import AsyncGenerator
from TTS.utils.synthesizer import Synthesizer

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService

class CoquiTTSService(TTSService):
    def __init__(self, *, model_path: str, config_path: str, use_cuda: bool = True, **kwargs):
        """
        Custom Coqui TTS Service for Pipecat.

        - Loads a **Vegeta fine-tuned** VITS model.
        - **Streams** audio output frame-by-frame for better responsiveness.
        - **Processes sentences as they arrive** (no waiting for full LLM response).

        Args:
            model_path (str): Path to the fine-tuned Coqui model.
            config_path (str): Path to the corresponding config.json file.
            use_cuda (bool): Whether to run inference on GPU (if available).
        """
        super().__init__(sample_rate=22050, **kwargs)
        self._use_cuda = use_cuda and torch.cuda.is_available()

        logger.info("🚀 Loading Coqui TTS Model...")
        self._synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            use_cuda=self._use_cuda
        )
        logger.info("✅ Coqui TTS Model Loaded!")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"🗣️ Generating TTS for: [{text}]")

        try:
            # 🔥 Generate waveform in a single pass
            wav = self._synthesizer.tts(text)

            # ✅ Ensure it's a NumPy array
            wav = np.array(wav, dtype=np.float32)

            # ✅ Normalize audio (scale between -1 and 1)
            wav = wav / max(abs(wav))

            # ✅ Resample from 22050 Hz → 16000 Hz
            target_sample_rate = 16000
            wav_tensor = torch.tensor(wav).unsqueeze(0)  # Convert to PyTorch tensor
            resampled_wav = torchaudio.transforms.Resample(orig_freq=22050, new_freq=target_sample_rate)(wav_tensor)
            wav = resampled_wav.squeeze().numpy()  # Convert back to NumPy

            # ✅ Convert to PCM S16LE (signed 16-bit int)
            wav = (wav * 32767).astype("int16")

            # ✅ Signal start of TTS
            yield TTSStartedFrame()

            # ✅ Stream audio in chunks (100ms chunks)
            chunk_size = target_sample_rate // 10  # 100ms chunks
            for i in range(0, len(wav), chunk_size):
                chunk = wav[i : i + chunk_size]
                yield TTSAudioRawFrame(audio=chunk, sample_rate=target_sample_rate, num_channels=1)
                await asyncio.sleep(0.1)  # Simulate streaming delay

            # ✅ Signal end of TTS
            yield TTSStoppedFrame()

        except Exception as e:
            logger.error(f"🔥 TTS Error: {e}")
            yield ErrorFrame(str(e))