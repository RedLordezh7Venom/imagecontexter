"""Category configuration and defaults.

Contains the default hardcoded classification categories:
  - anime: Anime, manga, animated Japanese art, anime characters
  - game: Video games, gaming screenshots, gameplay, game UI, 3D game models
  - movie: Live-action movies, TV shows, film stills, real actors, cinematic scenes
  - meme: Internet memes, humorous captioned photos, social media jokes, shitposts
  - coding: Code snippets, IDE/editor screenshots, terminal output, programming syntax

Users can also optionally load custom categories from a YAML file.
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


# Default hardcoded categories as requested
DEFAULT_CATEGORIES: list[Category] = [
    Category(
        name="anime",
        description="Anime, manga illustrations, animated Japanese characters, 2D anime art and styles",
    ),
    Category(
        name="game",
        description="Video games, gameplay screenshots, video game characters, gaming interfaces, 3D game engines",
    ),
    Category(
        name="movie",
        description="Live-action movies, TV series, cinematic scenes, film photography, real actors and actresses",
    ),
    Category(
        name="meme",
        description="Internet memes, image macros, funny captioned reaction images, humorous social media posts",
    ),
    Category(
        name="coding",
        description="Code editors, IDE screenshots, terminal/command line text, programming languages, syntax and code snippets",
    ),
]


@dataclass
class ClassifyConfig:
    """Classification configuration."""

    categories: list[Category]

    @classmethod
    def default(cls) -> ClassifyConfig:
        """Return the default hardcoded classification configuration."""
        return cls(categories=list(DEFAULT_CATEGORIES))

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
