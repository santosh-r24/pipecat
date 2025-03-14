#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from abc import ABC, abstractmethod
from typing import Optional


class BaseTextAggregator(ABC):
    @abstractmethod
    def aggregate(self, text: str) -> Optional[str]:
        pass

    @abstractmethod
    def handle_interruption(self):
        pass

    @abstractmethod
    def reset(self) -> str:
        pass
