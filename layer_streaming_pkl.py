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


def generate_labeled_edges(source):
    """
    Generate all legal moves from source.
    Returns:
        dict: {canonical_dest: set((pos, val))}
    """
    n = len(source)
    labeled_dests = defaultdict(set)

    for i in range(n):
        if source[i] != 0:
            continue

        left = source[(i - 1) % n]
        right = source[(i + 1) % n]

        for val in (1, -1):
            # local legality check
            if left == -val or right == -val:
                continue

            # now it's guaranteed legal
            new_board = list(source)
            new_board[i] = val

            dest = tuple(canonicalize_circular(new_board))
            labeled_dests[dest].add((i, val))

    return labeled_dests



def generate_edges_from_layer(layer):
    """
    Generate all labeled edges from a layer of canonical board states.

    For each source board in the given layer, this function applies all legal
    single-move placements (+1 or -1 in a zero position), canonicalizes the
    resulting board, and records the move(s) that produce each destination.

    Returns:
        dict:
            A nested dictionary of the form

                edges[source][dest] = set((pos, val))

            where:
              - source is a canonical board in the input layer
              - dest is a canonical board reachable from source in one move
              - (pos, val) indicates that placing val ∈ {+1, -1} at index pos
                produces dest (after canonicalization)

            Multiple distinct moves may map to the same destination due to
            rotational or sign symmetries.
    """
    edges = defaultdict(lambda: defaultdict(set))

    for source in layer:
        move_map = generate_labeled_edges(source)

        for dest, moves in move_map.items():
            edges[source][dest].update(moves)

    return {
        source: dict(dest_map)
        for source, dest_map in edges.items()
    }


def generate_game_graph_streaming(n, out_dir="graph_chunks"):
    """
    Memory-efficient, streaming game graph generation.

    Writes edge chunks to disk layer-by-layer instead of keeping
    the entire graph in memory.

    Each file contains:
        edges[source][dest] = set((pos, val))
    """
    import os
    out_dir=f"{n}_graph_chunks"
    os.makedirs(out_dir, exist_ok=True)

    prev_layer = set()

    for depth in reversed(range(n + 1)):
        print(f"Processing depth {depth}...")

        current_layer = set(boards_at_depth(n, depth))

        layer_edges = defaultdict(lambda: defaultdict(set))

        # Generate labeled edges
        raw_edges = generate_edges_from_layer(current_layer)

        for source, dest_map in raw_edges.items():
            for dest, moves in dest_map.items():
                if dest in prev_layer:
                    layer_edges[source][dest].update(moves)

        # Write this layer’s edges to disk
        path = f"{out_dir}/edges_depth_{depth}.pkl"
        with open(path, "wb") as f:
            pickle.dump(
                {src: dict(dests) for src, dests in layer_edges.items()},
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        # Free memory aggressively
        del layer_edges
        del raw_edges

        prev_layer = current_layer


# -------------------------------------------------
# To read back in the chunks:
# -------------------------------------------------
"""
import pickle
import glob

edges = {}

for path in sorted(glob.glob("graph_chunks/edges_depth_*.pkl")):
    with open(path, "rb") as f:
        chunk = pickle.load(f)
        edges.update(chunk)
"""
# ---------------------------------------------------------------------------------------------------------------
# Better: just stream chunks in during analysis, never load everything unless necessary
# ---------------------------------------------------------------------------------------------------------------


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
