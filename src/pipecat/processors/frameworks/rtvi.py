#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import dataclasses

from typing import List, Literal, Optional, Type
from pydantic import BaseModel, ValidationError

from pipecat.frames.frames import (
    BotInterruptionFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMModelUpdateFrame,
    StartFrame,
    SystemFrame,
    TTSSpeakFrame,
    TTSVoiceUpdateFrame,
    TranscriptionFrame,
    TransportMessageFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import AIService
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport

DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
    }
]

DEFAULT_MODEL = "llama3-70b-8192"

DEFAULT_VOICE = "79a125e8-cd45-4c13-8a67-188112f4dd22"


class RTVILLMConfig(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[dict]] = None


class RTVITTSConfig(BaseModel):
    voice: Optional[str] = None


class RTVIConfig(BaseModel):
    llm: Optional[RTVILLMConfig] = None
    tts: Optional[RTVITTSConfig] = None


class RTVISetup(BaseModel):
    config: Optional[RTVIConfig] = None


class RTVILLMMessageData(BaseModel):
    messages: List[dict]


class RTVITTSMessageData(BaseModel):
    text: str
    interrupt: Optional[bool] = False


class RTVIMessageData(BaseModel):
    setup: Optional[RTVISetup] = None
    config: Optional[RTVIConfig] = None
    llm: Optional[RTVILLMMessageData] = None
    tts: Optional[RTVITTSMessageData] = None


class RTVIMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: str
    data: Optional[RTVIMessageData] = None


class RTVIResponseData(BaseModel):
    type: str
    success: bool
    error: Optional[str] = None

class RTVIResponse(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["response"] = "response"
    data: RTVIResponseData


class RTVILLMContextMessageData(BaseModel):
    messages: List[dict]


class RTVIBotReady(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["bot-ready"] = "bot-ready"


class RTVILLMContextMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["llm-context"] = "llm-context"
    data: RTVILLMContextMessageData


class RTVITranscriptionMessageData(BaseModel):
    text: str
    user_id: str
    timestamp: str


class RTVITranscriptionMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["user-transcription"] = "user-transcription"
    data: RTVITranscriptionMessageData


class RTVIInterimTranscriptionMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["user-interim-transcription"] = "user-interim-transcription"
    data: RTVITranscriptionMessageData


class RTVIUserStartedSpeakingMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["user-started-speaking"] = "user-started-speaking"


class RTVIUserStoppedSpeakingMessage(BaseModel):
    label: Literal["realtime-ai"] = "realtime-ai"
    type: Literal["user-stopped-speaking"] = "user-stopped-speaking"


class RTVIProcessor(FrameProcessor):

    def __init__(
            self,
            *,
            transport: BaseTransport,
            setup: RTVISetup | None = None,
            llm_api_key: str = "",
            llm_base_url: str = "https://api.groq.com/openai/v1",
            tts_api_key: str = "",
            llm_cls: Type[AIService] = OpenAILLMService,
            tts_cls: Type[AIService] = CartesiaTTSService):
        super().__init__()
        self._transport = transport
        self._setup = setup
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._tts_api_key = tts_api_key
        self._llm_cls = llm_cls
        self._tts_cls = tts_cls
        self._start_frame: Frame | None = None
        self._llm: FrameProcessor | None = None
        self._tts: FrameProcessor | None = None
        self._pipeline: FrameProcessor | None = None

        self._frame_handler_task = self.get_event_loop().create_task(self._frame_handler())
        self._frame_queue = asyncio.Queue()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        else:
            await self._frame_queue.put((frame, direction))

        if isinstance(frame, StartFrame):
            self._start_frame = frame
            await self._handle_setup(self._setup)

    async def cleanup(self):
        self._frame_handler_task.cancel()
        await self._frame_handler_task

    async def _frame_handler(self):
        while True:
            try:
                (frame, direction) = await self._frame_queue.get()
                await self._handle_frame(frame, direction)
            except asyncio.CancelledError:
                break

    async def _handle_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, TransportMessageFrame):
            await self._handle_message(frame)
        else:
            await self.push_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) or isinstance(frame, InterimTranscriptionFrame):
            await self._handle_transcriptions(frame)
        elif isinstance(frame, UserStartedSpeakingFrame) or isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_interruptions(frame)

    async def _handle_transcriptions(self, frame: Frame):
        # TODO(aleix): Once we add support for using custom piplines, the STTs will
        # be in the pipeline after this processor. This means the STT will have to
        # push transcriptions upstream as well.

        message = None
        if isinstance(frame, TranscriptionFrame):
            message = RTVITranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp))
        elif isinstance(frame, InterimTranscriptionFrame):
            message = RTVIInterimTranscriptionMessage(
                data=RTVITranscriptionMessageData(
                    text=frame.text, user_id=frame.user_id, timestamp=frame.timestamp))

        if message:
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)

    async def _handle_interruptions(self, frame: Frame):
        message = None
        if isinstance(frame, UserStartedSpeakingFrame):
            message = RTVIUserStartedSpeakingMessage()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            message = RTVIUserStoppedSpeakingMessage()

        if message:
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)

    async def _handle_message(self, frame: TransportMessageFrame):
        try:
            message = RTVIMessage.model_validate(frame.message)
        except ValidationError as e:
            await self._send_response("setup", False, f"invalid message: {e}")
            return

        try:
            match message.type:
                case "setup":
                    setup = None
                    if message.data:
                        setup = message.data.setup
                    await self._handle_setup(setup)
                case "config-update":
                    await self._handle_config_update(message.data.config)
                case "llm-get-context":
                    await self._handle_llm_get_context()
                case "llm-append-context":
                    await self._handle_llm_append_context(message.data.llm)
                case "llm-update-context":
                    await self._handle_llm_update_context(message.data.llm)
                case "tts-speak":
                    await self._handle_tts_speak(message.data.tts)
                case "tts-interrupt":
                    await self._handle_tts_interrupt()

            # Send a message to indicate we successfully executed the command.
            await self._send_response(message.type, True)
        except ValidationError as e:
            await self._send_response(message.type, False, f"invalid message: {e}")

    async def _handle_setup(self, setup: RTVISetup | None):
        try:
            model = DEFAULT_MODEL
            if setup and setup.config and setup.config.llm and setup.config.llm.model:
                model = setup.config.llm.model

            messages = DEFAULT_MESSAGES
            if setup and setup.config and setup.config.llm and setup.config.llm.messages:
                messages = setup.config.llm.messages

            voice = DEFAULT_VOICE
            if setup and setup.config and setup.config.tts and setup.config.tts.voice:
                voice = setup.config.tts.voice

            self._tma_in = LLMUserResponseAggregator(messages)
            self._tma_out = LLMAssistantResponseAggregator(messages)

            self._llm = self._llm_cls(
                base_url=self._llm_base_url,
                api_key=self._llm_api_key,
                model=model)

            self._tts = self._tts_cls(api_key=self._tts_api_key, voice_id=voice)

            pipeline = Pipeline([
                self._tma_in,
                self._llm,
                self._tts,
                self._tma_out,
                self._transport.output(),
            ])
            self._pipeline = pipeline

            parent = self.get_parent()
            if parent and self._start_frame:
                parent.link(pipeline)

                # We need to initialize the new pipeline with the same settings
                # as the initial one.
                start_frame = dataclasses.replace(self._start_frame)
                await self.push_frame(start_frame)

            message = RTVIBotReady()
            frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
            await self.push_frame(frame)
        except Exception as e:
            await self._send_response("setup", False, f"unable to create pipeline: {e}")

    async def _handle_config_update(self, config: RTVIConfig):
        if config.llm and config.llm.model:
            frame = LLMModelUpdateFrame(config.llm.model)
            await self.push_frame(frame)
        if config.llm and config.llm.messages:
            frame = LLMMessagesUpdateFrame(config.llm.messages)
            await self.push_frame(frame)
        if config.tts and config.tts.voice:
            frame = TTSVoiceUpdateFrame(config.tts.voice)
            await self.push_frame(frame)

    async def _handle_llm_get_context(self):
        data = RTVILLMContextMessageData(messages=self._tma_in.messages)
        message = RTVILLMContextMessage(data=data)
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)

    async def _handle_llm_append_context(self, data: RTVILLMMessageData):
        if data and data.messages:
            frame = LLMMessagesAppendFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_llm_update_context(self, data: RTVILLMMessageData):
        if data and data.messages:
            frame = LLMMessagesUpdateFrame(data.messages)
            await self.push_frame(frame)

    async def _handle_tts_speak(self, data: RTVITTSMessageData):
        if data and data.text:
            if data.interrupt:
                await self._handle_tts_interrupt()
            frame = TTSSpeakFrame(text=data.text)
            await self.push_frame(frame)

    async def _handle_tts_interrupt(self):
        await self.push_frame(BotInterruptionFrame(), FrameDirection.UPSTREAM)

    async def _send_response(self, type: str, success: bool, error: str | None = None):
        # TODO(aleix): This is a bit hacky, but we might get invalid
        # configuration or something might going wrong during setup and we would
        # like to send the error to the client. However, if the pipeline is not
        # setup yet we don't have an output transport and therefore we can't
        # send any messages. So, we setup a super basic pipeline with just the
        # output transport so we can send messages.
        if not self._pipeline:
            pipeline = Pipeline([self._transport.output()])
            self._pipeline = pipeline

            parent = self.get_parent()
            if parent and self._start_frame:
                parent.link(pipeline)

        message = RTVIResponse(data=RTVIResponseData(type=type, success=success, error=error))
        frame = TransportMessageFrame(message=message.model_dump(exclude_none=True))
        await self.push_frame(frame)
