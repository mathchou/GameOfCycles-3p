from itertools import combinations, product
from collections import defaultdict
from main import is_legal, canonicalize_circular, legal_moves
import time
import tracemalloc
from pathlib import Path
import pickle

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


def generate_edges_from_layer(layer):
    """
    Generate edges from a layer of canonical boards by applying all legal moves.
    Returns a dict mapping canonical board -> list of canonical boards reachable in one move.
    """
    edges = defaultdict(list)

    for board in layer:
        # generate all legal moves from this board
        moves = legal_moves(board)
        for move in moves:
            move_canonical = canonicalize_circular(move)
            edges[board].append(move_canonical)

    return edges


def generate_game_graph_memory_efficient(n):
    """
    Memory-efficient game graph generation with deduplication after canonicalization.
    Edges are stored as sets to avoid duplicates.
    Streams layers from depth n down to 0.
    Only keeps two layers in memory at a time.

    Returns:
        edges_total: dict mapping canonical board -> set of canonical boards
    """
    edges_total = defaultdict(set)

    prev_layer = set()

    # Process layers from deepest to shallowest
    for depth in reversed(range(n + 1)):
        print(f"Processing depth {depth}...")
        raw_layer = boards_at_depth(n, depth)

        # Canonicalize and remove duplicates
        current_layer = set(canonicalize_circular(board) for board in raw_layer)

        # Generate edges from this layer to previous layer
        for board in current_layer:
            moves = legal_moves(board)
            for move in moves:
                move_canonical = canonicalize_circular(move)
                if move_canonical in prev_layer:
                    edges_total[board].add(move_canonical)  # store as set to remove duplicates

        # Current layer becomes previous layer for next iteration
        prev_layer = current_layer

    return edges_total


def save_graph(edges, filename):
    """Save edges dict as a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(edges, f)
    print(f"Graph saved to {filename}")

def run_graph_generation(max_n=10, mem_limit_mb=2000, output_dir="graphs"):
    """
    Progressively generate game graphs for n=1..max_n.
    Stops if estimated memory usage exceeds mem_limit_mb.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for n in range(1, max_n + 1):
        print(f"\n--- Generating game graph for n={n} ---")

        start_time = time.time()
        tracemalloc.start()

        try:
            edges = generate_game_graph_memory_efficient(n)
        except MemoryError:
            print(f"MemoryError: stopping at n={n}")
            break

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = time.time() - start_time

        # Stop if peak memory exceeds the limit
        if peak_mem / 1024**2 > mem_limit_mb:
            print(f"Peak memory {peak_mem/1024**2:.1f} MB exceeded limit of {mem_limit_mb} MB")
            break

        # Save graph
        filename = output_dir / f"game_graph_n{n}.pkl"
        save_graph(edges, filename)

        # Count unique nodes
        total_nodes = set()
        for src, dsts in edges.items():
            total_nodes.add(src)
            total_nodes.update(dsts)

        # Report stats
        print(f"Total nodes: {len(total_nodes)}")
        print(f"Time elapsed: {elapsed_time:.2f} s")
        print(f"Current memory: {current_mem / 1024**2:.2f} MB")
        print(f"Peak memory: {peak_mem / 1024**2:.2f} MB")

if __name__ == "__main__":
    run_graph_generation(max_n=25, mem_limit_mb=2000)
