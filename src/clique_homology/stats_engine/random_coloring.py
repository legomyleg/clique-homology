from random import choice
from typing import TypeAlias

Color: TypeAlias = str
DEFAULT_COLOR_PALETTE: list[Color] = ["RED", "BLUE", "GREEN", "YELLOW"]

def _validate_palette(allowed_colors: list[Color]) -> None:
    if not all(isinstance(color, str) for color in allowed_colors):
        raise TypeError("All allowed color values must be strings.")


def random_coloring(
    colors: list[Color],
    proportional: bool = False,
    allowed_colors: list[Color] | None = None,
) -> list[Color]:
    
    if not all(isinstance(color, str) for color in colors):
        raise TypeError("All input color values must be strings.")

    new_coloring: list[Color] = []
    palette: list[Color]

    if allowed_colors is not None:
        _validate_palette(allowed_colors)
        palette = allowed_colors
    else:
        palette = list(set(colors))

    if not palette and colors:
        raise ValueError("Color palette is empty but colors were provided.")
    
    if proportional:
        for _ in range(len(colors)):
            new_coloring.append(choice(palette))
    
    else:
        for _ in range(len(colors)):
            new_coloring.append(choice(palette))
            
    return new_coloring
