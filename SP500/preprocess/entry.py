from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from SP500.preprocess.base.reg import register_all
from SP500.preprocess.price.feat_price import PRICE_FEATURES
from SP500.preprocess.univ import register_sedol


@dataclass(frozen=True)
class RegisterTask:
    kind: str
    action: Callable[[], object]


TASKS: list[RegisterTask] = [
    RegisterTask("price", lambda: register_all(PRICE_FEATURES)),
    RegisterTask("sedol", register_sedol),
]


def register(kinds: Iterable[str] | None = None) -> None:
    wants = set(kinds) if kinds is not None else {t.kind for t in TASKS}
    known = {t.kind for t in TASKS}

    for task in TASKS:
        if task.kind in wants:
            task.action()

    unknown = wants - known
    if unknown:
        raise ValueError(f"Unsupported register kinds: {sorted(unknown)}")


__all__ = ["register", "RegisterTask", "TASKS"]
