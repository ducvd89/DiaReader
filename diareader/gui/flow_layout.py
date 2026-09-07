"""A layout that wraps its items onto as many rows as the width needs.

Qt ships nothing like this. QHBoxLayout puts everything on one line and lets the
widget grow past the screen edge, which is what the script editor's voice palette did
once the library reached seventeen voices: the chips ran off the right-hand side and
took the dialog with them. QGridLayout would wrap, but only at a column count fixed
when the widgets are added, so it is wrong at every width except the one it was
guessed for.

The interesting part of a wrapping layout is telling Qt that height depends on width.
A layout that only reports one size gets one row's worth of vertical space and clips;
``hasHeightForWidth`` plus ``heightForWidth`` is what makes the parent ask "how tall
at this width?" and reserve the answer.
"""
from __future__ import annotations

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy


class FlowLayout(QLayout):
    """Left-to-right, wrapping onto a new row when the current one is full."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # -- QLayout's required plumbing -------------------------------------------------

    def addItem(self, item) -> None:  # noqa: N802 — Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientations:  # noqa: N802
        # Nothing: the whole point is to be exactly as tall as the rows need. Claiming
        # vertical expansion here would let the palette eat the editor below it.
        return Qt.Orientations()

    # -- what makes it wrap ----------------------------------------------------------

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _layout(self, rect: QRect, apply: bool) -> int:
        """Place every item inside `rect`; return the total height used.

        Called twice for different reasons — once by heightForWidth to ask, once by
        setGeometry to tell — so `apply` decides whether the answer is also acted on.
        """
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x, y = effective.x(), effective.y()
        row_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if row_height > 0 and next_x - self._spacing > effective.right() + 1:
                # Does not fit on this row: drop to the next one and start again from
                # the left. row_height > 0 guards the case of a single item wider than
                # the whole layout, which must be placed rather than wrapped forever.
                x = effective.x()
                y = y + row_height + self._spacing
                next_x = x + hint.width() + self._spacing
                row_height = 0

            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))

            x = next_x
            row_height = max(row_height, hint.height())

        return y + row_height - rect.y() + margins.bottom()


def wrapping_row(widgets, spacing: int = 6) -> "QWidget":
    """A widget holding `widgets` in a FlowLayout, ready to drop into a QVBoxLayout.

    Layouts cannot be given a height-for-width contract by a parent layout alone —
    QVBoxLayout only honours it for widgets — so wrapping in one is what makes the
    rows actually reserve their space instead of being clipped to the first.
    """
    from PySide6.QtWidgets import QWidget

    holder = QWidget()
    policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    # Without this the layout's heightForWidth is never consulted by the parent, which
    # falls back to the height the rows would need at the *minimum* width — the widest
    # possible answer. That is not a cosmetic difference: the voice palette reported a
    # 384px minimum height where 98px was what it actually used, the non-verbal box
    # 533px against 104px, and between them they gave the dialog a 1107px minimum
    # height that pushed its lower half off the screen.
    policy.setHeightForWidth(True)
    holder.setSizePolicy(policy)
    layout = FlowLayout(holder, spacing=spacing)
    for widget in widgets:
        layout.addWidget(widget)

    # One row is the floor, stated explicitly because Qt derives its own from
    # heightForWidth *at the layout's minimum width* — which for a flow layout is the
    # width of a single item, so it answers with every item stacked in one column: 384px
    # for the voice palette, 533px for the non-verbal box. An explicit minimum takes
    # precedence over that and lets the window shrink; the layout still asks for its
    # real height through heightForWidth whenever there is room to give it.
    if widgets:
        row_height = max(w.sizeHint().height() for w in widgets)
        margins = layout.contentsMargins()
        holder.setMinimumHeight(row_height + margins.top() + margins.bottom())
    return holder
