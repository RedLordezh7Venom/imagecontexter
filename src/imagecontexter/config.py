"""Category configuration loader.

Reads a YAML file defining the classification categories and their
descriptions. The descriptions are fed to the VLM as context so it
understands what each category means.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Category:
    """A single classification category."""

    name: str
    description: str = ""

    def __str__(self) -> str:
        if self.description:
            return f"{self.name}: {self.description}"
        return self.name


@dataclass
class ClassifyConfig:
    """Complete classification configuration parsed from a YAML file.

    Expected YAML format::

        categories:
          - name: landscapes
            description: "Outdoor nature scenes ..."
          - name: people
            description: "Photos with humans ..."
    """

    categories: list[Category]

    # --- factory ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClassifyConfig:
        """Load categories from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Categories file not found: {path}")

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "categories" not in data:
            raise ValueError(
                f"Invalid categories file: {path}. "
                "Expected a YAML file with a top-level 'categories' key."
            )

        raw_cats = data["categories"]
        if not raw_cats:
            raise ValueError("At least one category must be defined.")

        categories: list[Category] = []
        for entry in raw_cats:
            if isinstance(entry, str):
                # Allow shorthand: just a name
                categories.append(Category(name=entry))
            elif isinstance(entry, dict):
                categories.append(
                    Category(
                        name=entry["name"],
                        description=entry.get("description", ""),
                    )
                )
            else:
                raise ValueError(f"Invalid category entry: {entry!r}")

        return cls(categories=categories)

    # --- helpers ----------------------------------------------------------

    def category_names(self) -> list[str]:
        """Return a list of all category names."""
        return [c.name for c in self.categories]

    def find_category(self, name: str) -> Category | None:
        """Find a category by name (case-insensitive)."""
        name_lower = name.lower()
        for cat in self.categories:
            if cat.name.lower() == name_lower:
                return cat
        return None

    def has_other(self) -> bool:
        """Check whether an 'other' catch-all category is defined."""
        return self.find_category("other") is not None
