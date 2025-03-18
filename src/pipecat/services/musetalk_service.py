#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import numpy as np
import torch
import cv2
import base64
from loguru import logger
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import process_audio_frame
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VideoFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService

# Load MuseTalk models at initialization
logger.info("🚀 Loading MuseTalk models...")
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
logger.info("✅ MuseTalk models loaded successfully!")

class MuseTalkService(AIService):
    """Pipecat Service to generate animated video frames using MuseTalk."""

    def __init__(
        self,
        base_video_path: str,
        output_resolution=(512, 512),
        fps=25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_video_path = base_video_path
        self.output_resolution = output_resolution
        self.fps = fps
        self.frame_buffer = []
        self.audio_buffer = []
        self._load_base_video()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_base_video(self):
        """Preload the base head-moving video for the avatar."""
        cap = cv2.VideoCapture(self.base_video_path)
        self.base_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.output_resolution)
            self.base_frames.append(frame)
        cap.release()
        logger.info(f"✅ Loaded {len(self.base_frames)} frames from base video.")

    def _generate_video_frame(self, audio_chunk):
        """Generate a single frame using MuseTalk for the given audio chunk."""
        audio_features = process_audio_frame(audio_chunk)
        audio_features = torch.from_numpy(audio_features).to(self.device).unsqueeze(0)
        audio_features = pe(audio_features)

        base_frame = self.base_frames[len(self.frame_buffer) % len(self.base_frames)]
        base_frame_resized = cv2.resize(base_frame, (256, 256))
        latents = vae.get_latents_for_unet(base_frame_resized)

        pred_latents = unet.model(latents, timesteps, encoder_hidden_states=audio_features).sample
        generated_frame = vae.decode_latents(pred_latents)

        return generated_frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and generate animated responses."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            logger.info("🟢 MuseTalk Animation Started")
            self.frame_buffer = []
            self.audio_buffer = []

        elif isinstance(frame, TTSAudioRawFrame):
            logger.debug("🔊 Received Audio Chunk for MuseTalk")
            self.audio_buffer.append(frame.audio)
            video_frame = self._generate_video_frame(frame.audio)
            self.frame_buffer.append(video_frame)

            encoded_frame = base64.b64encode(cv2.imencode('.jpg', video_frame)[1]).decode("utf-8")
            video_response_frame = VideoFrame(video_data=encoded_frame)

            await self.push_frame(video_response_frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            logger.info("🔴 MuseTalk Animation Stopped")
            self.frame_buffer.clear()
            self.audio_buffer.clear()

        else:
            await self.push_frame(frame, direction)
