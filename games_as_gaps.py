from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Gap:
    size: int
    orientation: int   # +1 or -1

    def is_dead(self) -> bool:
        # Negative gap of size 1 is unplayable
        return self.orientation == -1 and self.size == 1

    def legal_moves(self) -> List[Tuple["Gap", ...]]:
        moves = []

        if self.size <= 0 or self.is_dead():
            return moves

        # ---- SHRINK (remove boundary zero) ----
        new_size = self.size - 1
        if new_size > 0:
            moves.append((Gap(new_size, self.orientation),))
        else:
            moves.append(tuple())  # gap disappears

        # ---- SPLIT (fill interior zero) ----
        if self.size >= 3:
            # interior positions: a >= 1, b >= 1 with a + b = size - 1
            for a in range(1, self.size - 1):
                b = self.size - 1 - a
                # type pairs according to orientation
                type_pairs = [(1,1), (-1,-1)] if self.orientation == 1 else [(1,-1), (-1,1)]
                for tL, tR in type_pairs:
                    moves.append((Gap(a, tL), Gap(b, tR)))

        return moves

    def __repr__(self) -> str:
        sign = '+' if self.orientation == 1 else '-'
        return f"{sign}{self.size}"

    def __str__(self) -> str:
        return self.__repr__()


class GameState:
    def __init__(self, gaps: List[Gap], turn: int = 0):
        # Convert tuples to Gap automatically
        gaps_list = [g if isinstance(g, Gap) else Gap(*g) for g in gaps]
        # Sort gaps immediately in canonical order
        self.gaps = sorted(gaps_list, key=lambda g: (-g.size, -g.orientation))
        self.turn = turn % 3

    def canonical(self) -> Tuple[Tuple[int,int], ...]:
        """Return canonical hashable representation for memoization."""
        return tuple((g.size, g.orientation) for g in self.gaps)

    def legal_moves(self) -> List["GameState"]:
        """Return list of unique successor states (duplicate-free)."""
        next_states_set = set()
        next_states_list = []

        for i, gap in enumerate(self.gaps):
            for move in gap.legal_moves():
                new_gaps = self.gaps[:i] + list(move) + self.gaps[i+1:]
                new_state = GameState(new_gaps, self.turn + 1)
                key = (new_state.canonical(), new_state.turn)
                if key not in next_states_set:
                    next_states_set.add(key)
                    next_states_list.append(new_state)

        return next_states_list

    def is_terminal(self) -> bool:
        """Game is over if no gaps have legal moves."""
        return all(not gap.legal_moves() for gap in self.gaps)

    def __repr__(self) -> str:
        gaps_str = ', '.join(str(g) for g in self.gaps)
        return f"[{gaps_str}] (Player {self.turn + 1}'s turn)"

    def __str__(self) -> str:
        return self.__repr__()
