from dataclasses import dataclass, field
from typing import List


@dataclass()
class Results():
    pnsr: List[int] = field(default_factory=list)
