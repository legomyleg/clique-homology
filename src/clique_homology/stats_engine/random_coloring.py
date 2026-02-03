from typing import Literal
from random import choice

Color = Literal["RED", "BLUE", "GREEN", "YELLOW"]

def random_coloring(colors: list[Color], proportional:bool=False) -> list[Color]: # type: ignore
    
    new_coloring: list[Color] = []
    
    if proportional:
        for i in range(len(colors)):
            new_coloring.append(choice(colors))
    
    else:
        used_colors: set[Color] = set()
        
        for color in colors:
            used_colors.add(color)
        
        for _ in range(len(colors)):
            new_coloring.append(choice(list(used_colors)))
            
    return new_coloring