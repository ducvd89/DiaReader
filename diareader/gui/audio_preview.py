"""Playing one cached sentence back, inside the app.

Small on purpose. The question a listener asks in the review table is "is this take
right" — press, hear, decide — so there is no transport, no seeking and no playlist,
and pressing play on a second sentence stops the first rather than overlapping it.

QtMultimedia rather than an audio library: PySide6 already ships it, the cached takes
are ordinary PCM WAVs on disk, and adding a dependency to play a file the app just
wrote would be a poor trade.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

log = logging.getLogger(__name__)


class AudioPreview(QObject):
    """One player, shared by every row of the table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player: QMediaPlayer | None = None
        self._output: QAudioOutput | None = None

    def _ensure(self) -> QMediaPlayer:
        if self._player is None:
            # Held on the instance, not made locally: an output that goes out of scope
            # takes the sound with it and the file plays silently.
            self._output = QAudioOutput()
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._output)
            self._player.errorOccurred.connect(
                lambda _error, text: log.warning("Could not play the take: %s", text)
            )
        return self._player

    def play(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            log.warning("No audio at %s", path)
            return
        player = self._ensure()
        player.stop()
        player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        player.play()

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()
