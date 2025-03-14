#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from abc import ABC
from typing import Optional

from loguru import logger

from pipecat.utils.utils import obj_count, obj_id


class BaseObject(ABC):
    def __init__(self, *, name: Optional[str] = None):
        self._id: int = obj_id()
        self._name = name or f"{self.__class__.__name__}#{obj_count(self)}"
        self._event_handlers: dict = {}

        self._scheduled_tasks = set()

        self._scheduler_queue = asyncio.Queue()
        self._scheduler_task = asyncio.create_task(self._scheduler_task_handler())

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    async def cleanup(self):
        if self._scheduled_tasks:
            event_names, tasks = zip(*self._scheduled_tasks)
            logger.debug(f"{self} wating on event handlers to finish {list(event_names)}...")
            await asyncio.wait(tasks)

    def event_handler(self, event_name: str):
        def decorator(handler):
            self.add_event_handler(event_name, handler)
            return handler

        return decorator

    def add_event_handler(self, event_name: str, handler):
        if event_name not in self._event_handlers:
            raise Exception(f"Event handler {event_name} not registered")
        self._event_handlers[event_name].append(handler)

    def _register_event_handler(self, event_name: str):
        if event_name in self._event_handlers:
            raise Exception(f"Event handler {event_name} already registered")
        self._event_handlers[event_name] = []

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        await self._scheduler_queue.put((event_name, args, kwargs))

    async def _run_task(self, event_name: str, *args, **kwargs):
        try:
            for handler in self._event_handlers[event_name]:
                if inspect.iscoroutinefunction(handler):
                    await handler(self, *args, **kwargs)
                else:
                    handler(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in event handler {event_name}: {e}")

    def _task_finished(self, task: asyncio.Task):
        tuple_to_remove = next((t for t in self._scheduled_tasks if t[1] == task), None)
        if tuple_to_remove:
            self._scheduled_tasks.discard(tuple_to_remove)

    async def _scheduler_task_handler(self):
        while True:
            (event_name, args, kwargs) = await self._scheduler_queue.get()

            # Create the task.
            task = asyncio.create_task(self._run_task(event_name, *args, **kwargs))

            # Add it to our list of event tasks.
            self._scheduled_tasks.add((event_name, task))

            # Remove the task from the event tasks list when the task completes.
            task.add_done_callback(self._task_finished)

            self._scheduler_queue.task_done()

    def __str__(self):
        return self.name
