"""DiaReader GUI entry point."""
from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from diareader import config, no_console
from diareader.gui.main_window import MainWindow


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # First, before anything can spawn a process: torch.compile runs a C compiler per
    # kernel and each one would otherwise pop a console window over whatever the user
    # is doing. Applies to --selftest too, which compiles as part of its checks.
    no_console.hide_child_consoles()

    # `DiaReader.exe --selftest` is how the packaged build is diagnosed. It has no
    # console, so every failure — wrong torch wheel, missing FFmpeg, no token — looks
    # identical from outside: the window does not appear, or narration comes out
    # silent. See diareader/selftest.py.
    if "--selftest" in sys.argv[1:]:
        from diareader.selftest import main as selftest_main

        sys.exit(selftest_main())

    config.ensure_dirs()

    # Before the window reads the settings, so voices shipped with the build are
    # already in the list rather than appearing only after a restart.
    from diareader import bundled_voices

    bundled_voices.install_if_new(config.Settings.load())

    # Windows groups taskbar buttons by "app id", and without one of our own it
    # decides this process belongs to whatever launched it — so the taskbar shows
    # the host's icon rather than the window's. Must be set before any window exists.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("DiaReader.App")
        except Exception:  # noqa: BLE001 — cosmetic; never worth failing startup
            pass

    app = QApplication(sys.argv)
    # On the application, not the window: dialogs and message boxes inherit it, so
    # the icon follows every window the app opens rather than just the main one.
    icon_path = config.asset_path("icon.ico")
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        logging.getLogger(__name__).warning("Application icon missing at %s", icon_path)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
