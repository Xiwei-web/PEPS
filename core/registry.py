"""Small registries used by schema libraries and tool collections."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from peps.core.enums import PrimitiveFamily
from peps.core.exceptions import RegistryError
from peps.core.types import PrimitiveDefinition

T = TypeVar("T")


@dataclass(slots=True)
class Registry(Generic[T]):
    """A typed name-to-object registry with duplicate protection."""

    name: str
    key_fn: Callable[[T], str]
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, item: T, *, overwrite: bool = False) -> T:
        key = self.key_fn(item)
        if not key:
            raise RegistryError(f"{self.name} registry cannot register an empty key")
        if key in self._items and not overwrite:
            raise RegistryError(f"{self.name} registry already contains key: {key}")
        self._items[key] = item
        return item

    def register_many(self, items: Iterable[T], *, overwrite: bool = False) -> None:
        for item in items:
            self.register(item, overwrite=overwrite)

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            raise RegistryError(f"{self.name} registry missing key: {key}") from exc

    def maybe_get(self, key: str) -> T | None:
        return self._items.get(key)

    def contains(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return list(self._items.keys())

    def values(self) -> list[T]:
        return list(self._items.values())

    def items(self) -> list[tuple[str, T]]:
        return list(self._items.items())

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


class PrimitiveRegistry(Registry[PrimitiveDefinition]):
    """Registry specialized for persistent primitive definitions."""

    def __init__(self, name: str = "primitive") -> None:
        super().__init__(name=name, key_fn=lambda item: item.id)

    def by_family(self, family: PrimitiveFamily | str) -> list[PrimitiveDefinition]:
        family = PrimitiveFamily.coerce(family)
        return [item for item in self.values() if item.family is family]

    def by_tag(self, tag: str) -> list[PrimitiveDefinition]:
        return [item for item in self.values() if tag in item.tags]

    def definition_ids(self) -> set[str]:
        return set(self.keys())

