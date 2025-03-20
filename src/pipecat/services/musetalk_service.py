#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import os 
import sys
import asyncio
import numpy as np
import torch
import cv2
import base64
from loguru import logger

# ✅ Ensure MuseTalk is discoverable
MUSETALK_PATH = "/content/drive/MyDrive/VegetaAvatar/MuseTalk"
if MUSETALK_PATH not in sys.path:
    sys.path.append(MUSETALK_PATH)

# ✅ Import MuseTalk Core Functions
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox

# ✅ Import Pipecat Core Components
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    OutputImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService


class MuseTalkService(AIService):
    """Pipecat Service to generate animated video frames using MuseTalk."""

    def __init__(self, transport, base_video_path: str, **kwargs):
        super().__init__(**kwargs)
        self.transport = transport  # ✅ Store Pipecat Transport
        self.fps = self.transport.get_fps()  # ✅ Fetch FPS dynamically

        # 🎥 Load MuseTalk Models (Preloaded for Performance)
        logger.info("🚀 Loading MuseTalk models...")
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)
        logger.info("✅ MuseTalk models loaded successfully!")

        # 🎥 Load Base Video (Head Movement)
        self.base_video_path = base_video_path
        self.frames, self.coords = get_landmark_and_bbox([self.base_video_path])
        self.latents = [self.vae.get_latents_for_unet(cv2.resize(frame, (256, 256))) for frame in self.frames]

        self.idx = 0  # Frame tracking index
        self.tts_active = False  # ✅ Track whether speech is happening

        # ✅ Start looping base video on launch
        asyncio.create_task(self.loop_base_video())

    def can_generate_metrics(self) -> bool:
        return True

    def process_audio_frame(self, audio_chunk):
        """
        Processes an incoming audio chunk to extract MuseTalk features.
        """
        audio_feature = self.audio_processor.audio2feat(audio_chunk)
        return self.audio_processor.feature2chunks(feature_array=audio_feature, fps=self.fps)  # ✅ Dynamic FPS handling

    def get_next_latent(self):
        """
        Fetch the next precomputed latent vector for MuseTalk generation.
        """
        latent = self.latents[self.idx % len(self.latents)]
        self.idx += 1
        return latent.to(dtype=self.unet.model.dtype)

    def generate_musetalk_frame(self, base_frame, audio_feature):
        """
        Generates a lip-synced video frame using MuseTalk.
        """
        latent_batch = self.get_next_latent()  # ✅ Use precomputed latent
        audio_feature = torch.from_numpy(audio_feature).to(self.unet.device, dtype=self.unet.model.dtype)
        pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature).sample
        recon_frame = self.vae.decode_latents(pred_latents)[0]  # Convert latent back to image

        # Resize & overlay onto base frame
        lip_sync_frame = cv2.resize(recon_frame.astype(np.uint8), (base_frame.shape[1], base_frame.shape[0]))
        return lip_sync_frame  # Returns the animated frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Handles incoming audio frames, generates a talking avatar, and outputs video frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            logger.info("🟢 MuseTalk Animation Started")
            await self.start_ttfb_metrics()
            self.idx = 0  # Reset frame index when speech starts
            self.tts_active = True  # ✅ Set TTS as active

        elif isinstance(frame, TTSAudioRawFrame):
            logger.debug("🔊 Received Audio Chunk for MuseTalk")
            # 🎤 Process the incoming audio frame for MuseTalk
            processed_feature = self.process_audio_frame(frame.audio)

            # 🎥 Generate a lip-synced video frame
            if self.idx < len(self.frames):  # 🔄 Prevent out-of-bounds
                base_frame = self.frames[self.idx % len(self.frames)]
                video_frame = self.generate_musetalk_frame(base_frame, processed_feature)

                # 🖼️ Convert video frame to Base64 format
                encoded_frame = base64.b64encode(cv2.imencode(".jpg", video_frame)[1]).decode("utf-8")

                # 🚀 Push the generated image frame (instead of video)
                await self.push_frame(OutputImageRawFrame(image=encoded_frame, size=(video_frame.shape[1], video_frame.shape[0]), format="jpeg"))

        elif isinstance(frame, TTSStoppedFrame):
            await self.stop_ttfb_metrics()
            self.idx = 0  # Reset frame index when speech stops
            self.tts_active = False  # ✅ Mark TTS as inactive

            # 🔄 Start looping base video when there's no speech
            asyncio.create_task(self.loop_base_video())

    async def loop_base_video(self):
        """Loop Base Video When No Speech is Detected."""
        logger.info("🔁 No Speech Detected: Playing Base Video Loop")

        while not self.tts_active:
            base_frame = self.frames[self.idx % len(self.frames)]
            encoded_frame = base64.b64encode(cv2.imencode(".jpg", base_frame)[1]).decode("utf-8")
            await self.push_frame(OutputImageRawFrame(image=encoded_frame, size=(base_frame.shape[1], base_frame.shape[0]), format="jpeg"))
            self.idx = (self.idx + 1) % len(self.frames)

            await asyncio.sleep(1 / self.fps)  # ✅ Ensure video plays at correct FPS
