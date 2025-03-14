#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class SimpleTextAggregator(BaseTextAggregator):
    def __init__(self):
        self._text = ""

    def aggregate(self, text: str) -> Optional[str]:
        result: Optional[str] = None

        self._text += text

        eos_end_marker = match_endofsentence(self._text)
        if eos_end_marker:
            result = self._text[:eos_end_marker]
            self._text = self._text[eos_end_marker:]

        return result

    def handle_interruption(self):
        self._text = ""

    def reset(self) -> str:
        text = self._text
        self._text = ""
        return text
