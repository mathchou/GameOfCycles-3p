from itertools import combinations, product
from collections import defaultdict
from main import is_legal, canonicalize_circular

def is_partial_legal(board, i):
    # no adjacent different nonzeros
    if i > 0 and board[i] != 0 and board[i] + board[i - 1] == 0:
        return False
    return True


def boards_at_depth(n, depth):
    boards = set()
    board = [0] * n

    # canonical anchor
    if depth > 0:
        board[0] = 1
        start_pos = 1
        remaining = depth - 1
    else:
        return {tuple(board)}

    def fill_intelligently(pos, remaining):
        # impossible to finish
        if remaining > n - pos:
            return

        if pos == n:
            if remaining == 0 and is_legal(board):
                boards.add(tuple(canonicalize_circular(board)))
                #boards.add(tuple(board)) #first just add the board so we can see what's happening
            return

        # option 1: fill in zero, doesn't change remaining non-zeros
        if n - pos > remaining: # only run this if there are more positions left than remaining +-1's to add
            board[pos] = 0
            fill_intelligently(pos + 1, remaining)
            # so, the first boards it will look at will be filling 0's until the last spots are force to be non-zero

        # option 2: fill in +1 or -1, but don't make it negative the previous (illegal board)
        if remaining > 0:
            for v in (1, -1):
                if board[pos-1] == 0 or board[pos-1] == v: # only continues if we aren't creating a 1/-1 or -1/1 sequence
                    board[pos] = v
                    if is_partial_legal(board, pos): # this is now only here for a sanity check, our rules should already guarantee this
                        fill_intelligently(pos + 1, remaining - 1)

    fill_intelligently(start_pos, remaining)
    return boards


def generate_edges(layer_lower, layer_upper):
    """
    Generate edges from boards at depth k-1 to boards at depth k.
    layer_lower: set of canonical boards at depth k-1
    layer_upper: set of canonical boards at depth k
    Returns: dict mapping canonical lower -> list of canonical upper
    WRONG
    """
    edges = defaultdict(list)

    for lower in layer_lower:
        for upper in layer_upper:
            # check value difference to detect new non-zero
            diff_indices = [i for i in range(len(lower)) if lower[i] != upper[i]]

            if len(diff_indices) != 1:
                continue  # must differ at exactly one index

            idx = diff_indices[0]
            # ensure this is a 0 -> ±1 change
            if lower[idx] != 0 or upper[idx] == 0:
                continue

            edges[lower].append(upper)

    return edges




if __name__ == "__main__":
    n = 5
    for k in range(n+1):
        print(boards_at_depth(n, k))

    print(generate_edges(boards_at_depth(5,2), boards_at_depth(5,3)))