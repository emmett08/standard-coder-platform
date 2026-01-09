from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass(frozen=True)
class Ui:
    console: Console
    progress: Progress

    def log(self, message: str) -> None:
        self.console.print(message)


@contextmanager
def progress_ui() -> Iterator[Ui]:
    console = Console()
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    with progress:
        yield Ui(console=console, progress=progress)
