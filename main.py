"""
Calculation of the degree of nontrivial ultrametricity for RNA macrostates.
PHYSICALLY RIGOROUS APPROACH: distance between basins via spectral
decomposition of the transition rate matrix (Mahalanobis distance
in the space of eigenvectors of the symmetrized K matrix).

METHOD:
1. Construct the transition rate matrix K between all structures
   (N x N, where N ~ 2000) based on the Kramers formula.
2. Symmetrize K taking into account detailed balance.
3. Compute m smallest (by absolute value) eigenvalues
   and corresponding eigenvectors (Lanczos method for
   sparse matrices).
4. Perform automatic filtering of noise modes by searching for
   a spectral gap: if the ratio |λ_k| / |λ_{k-1}| exceeds
   a threshold (default 10^6), modes with indices < k are discarded
   as numerical noise.
5. Each basin of attraction is represented by a characteristic
   vector χ_A in the space of structures.
6. The distance between basins A and B is defined as the weighted
   Euclidean distance between the projections of χ_A and χ_B onto
   the eigenvectors (Mahalanobis distance).
7. The resulting distance matrix is a metric and is checked
   for ultrametricity.

PROCESSING DISCONNECTED GRAPHS:
Before constructing the K_sym matrix, the connectivity of the structure graph is checked.
If the graph contains several connected components, each component
is processed separately: its own K_sym matrix is built,
spectral decomposition and ultrametricity check are performed.
Components with fewer than 3 basins are skipped.

STATISTICAL MODE (NUM_STAT > 1):
When NUM_STAT > 1, NUM_STAT independent runs with different random
structure samples are performed for each sequence (seed varies:
RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
Results are averaged, and the final table displays mean values
and standard deviations (mean ± std). Integer quantities
(number of structures, basins, connected components) are rounded to integers.

OUTPUT MODES:
VERBOSE = True  — full log (steps, components, spectral analysis).
VERBOSE = False — brief log: sequence header and parameters
                  are printed once, then only RUN/COMPLETED,
                  then the statistics block.

ADVANTAGES:
- Takes into account all possible transition paths (via spectral decomposition).
- Context-independent (distance between A and B is determined only by
  them, not by the presence of other basins).
- Symmetric and is guaranteed to be a metric.
- Automatically filters numerical noise by searching for a spectral gap.
- Correctly handles disconnected structure graphs.
- Computational complexity O(m·N·E + K²·m), allowing processing of
  N ~ 2000 structures and K ~ 100 basins in seconds.

STRUCTURE GENERATION MODE:
Stochastic sampling (pbacktrack) from the Gibbs distribution.

Dependencies: pip install viennarna numpy scipy biopython
"""

import RNA
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations
from collections import defaultdict
import warnings
import time
import textwrap
import os
import glob
import gc
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL VARIABLES FOR PARALLEL NEIGHBOR GENERATION
# ============================================================================

_ALLOWED_PAIRS = None
"""Global variable: list of allowed pairs [(i, j, mask), ...]."""

_PAIR_POS_MASK_MAP = None
"""Global variable: dictionary (i,j) -> mask for fast retrieval of pair mask."""

_SEQ_LEN = None
"""Global variable: sequence length."""

# ============================================================================
# USER PARAMETERS
# ============================================================================

# --- Data Source Parameters ---

FASTA_RNA = True
"""
Mode for loading RNA sequences from FASTA files.
  True  — scan the current folder for *.fasta files,
          load all sequences, sort by length.
  False — use the sequence from RNA_SEQUENCE.
Recommended value: True (for research work).
"""

RNA_SEQUENCE = "ACATCAATCCACCACTCTTTCTCTTTAAAAAGAGTAGACCCAGGAACCGAAATTCTTTACCAAATTAAAAAA"
"""
Primary RNA structure. Used only when FASTA_RNA = False.
Allowed characters: A, U, G, C (uppercase, T is automatically replaced with U).
Recommended length: 50–200 nucleotides.
"""

# --- Temperature and Energy Parameters ---

TEMPERATURE_CELSIUS = 37.0
"""
Temperature in degrees Celsius.
Affects Boltzmann weights and transition probabilities.
  Low (< 20°C): deep basins, rare transitions.
  High (> 60°C): smoothed landscape, fast transitions.
Recommended value: 37.0 (physiological temperature).
Allowed range: 0.0 – 100.0.
"""

ENERGY_WINDOW = 100.0
"""
Energy window (kcal/mol) relative to the minimum free energy (MFE).
During stochastic sampling, structures with energy > MFE + ENERGY_WINDOW
are discarded. If set to "inf" — no window restriction.
  Small window (1–5 kcal/mol): only the most stable structures,
    may be insufficient for analysis.
  Large window (> 15 kcal/mol): many structures, sparse graph,
    computation time increases.
Recommended value: 10.0.
Allowed range: positive number or "inf".
"""

# --- Structure Generation Parameters ---

MAX_STRUCTURES = 100000
"""
Maximum number of generated secondary structures (microstates).
Generation stops when the number of unique structures within the specified
energy window reaches this value.
  Small (100–500): fast but statistically poor analysis.
  Large (> 10000): complete landscape picture, but slow.
Recommended value: 2000–5000.
Allowed range: 100 – 20000.
"""

MIN_HAIRPIN_LEN = 3
"""
Minimum number of unpaired nucleotides in a hairpin loop.
Defines the condition: j - i - 1 >= MIN_HAIRPIN_LEN.
  Standard value: 3 (steric restriction).
  Value 0 disables the restriction (non-physical).
Recommended value: 3.
Allowed range: 0 – 10.
"""

RANDOM_SEED = 42
"""
Initial value for the random number generator.
Ensures reproducibility of results.
When NUM_STAT > 1, seed varies: RANDOM_SEED, RANDOM_SEED+1, ...
Recommended value: 42 (or any integer).
Allowed range: any integer.
"""

# --- Basin of Attraction Parameters ---

MAX_MACROSTATES_ANALYSIS = 500
"""
Maximum number of attraction basins participating in the final analysis.
If more remain after filtering, basins with the largest
statistical sums Z are kept.
  Small (10–30): fast, but may be insufficient for triplet statistics.
  Large (> 200): more triplets for analysis, but slower (K³ for spectrum).
Recommended value: 100.
Allowed range: 3 – 500.
"""

MIN_MACROSTATE_SIZE = 5
"""
Minimum size of an attraction basin (number of structures it contains).
Basins smaller than this are considered statistically insignificant.
  Value 1: all basins are included, including isolated structures.
  Value 5–10: small artifact basins are filtered out.
Recommended value: 5.
Allowed range: 1 – 100.
"""

# --- Spectral Analysis Parameters ---

NUM_EIGENMODES = 50
"""
Number of eigenmodes (eigenvalues and eigenvectors) requested
for spectral decomposition. After automatic filtering of noise modes,
the actual number of used modes may be smaller.
  Small (5–10): fast, but information about fine landscape structure is lost.
  Large (> 100): more accurate, but slower (scales linearly).
  Constraint: must be strictly less than the number of structures.
Recommended value: 50.
Allowed range: 5 – 200 (but not more than N-2, where N is the number of structures).
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for detecting a spectral gap between noise and physical modes.
If the ratio |λ_k| / |λ_{k-1}| > SPECTRAL_GAP_THRESHOLD, modes with indices
< k are considered numerical noise and are discarded.
  Large threshold (10^8): conservative, may lose weak physical modes.
  Small threshold (10^2): aggressive, may retain noise modes.
Recommended value: 1e6.
Allowed range: 1e2 – 1e12.
"""

FREQUENCY_PREFACTOR = 1.0
"""
Frequency factor ν₀ in the Kramers formula (in arbitrary units).
Affects the absolute scale of the K matrix, but does not affect eigenvectors
or relative distances between basins (changing ν₀ multiplies all λ_k
by a constant, which cancels out in the Mahalanobis distance).
Recommended value: 1.0 (leave unchanged).
Allowed range: any positive number.
"""

EIGS_MAXITER = 50000
"""
Maximum number of iterations for the Lanczos algorithm (ARPACK) when computing
eigenvalues of the K_sym matrix. Increasing this parameter improves
convergence for matrices with a dense spectrum near zero, but increases
computation time.
Recommended value: 50000.
Allowed range: 1000 – 200000.
"""

EIGS_SIGMA = 1e-10
"""
Shift sigma for the Lanczos algorithm when searching for eigenvalues
near zero. The value must be positive and sufficiently small
not to distort the spectrum of physical modes (which have |λ| >= 10^-4),
but sufficiently large to avoid numerical singularity when
solving the system (K_sym - sigma*I)x = b.
  Too small (10^-15): risk of numerical singularity.
  Too large (10^-3): distorts the spectrum.
Recommended value: 1e-10.
Allowed range: 1e-12 – 1e-6.
"""

# --- Ultrametricity Check Parameters ---

ULTRAMETRIC_EPSILON = 0.05
"""
Relative accuracy ε for checking approximate ultrametricity.
Two largest sides of a triangle are considered equal if
(d_max - d_mid) / d_mid <= ε.
  Must be strictly less than ULTRAMETRIC_DELTA.
  At ε = 0: exact equality is required (almost unattainable).
  At ε > 0.1: many false-positive classifications.
Recommended value: 0.05.
Allowed range: 0.0 – 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference δ between the smaller and middle sides
of a triangle for classification as nontrivially ultrametric:
(d_mid - d_min) / d_mid > δ.
  Must be strictly greater than ULTRAMETRIC_EPSILON.
  At small δ: equilateral triangles are erroneously classified
    as nontrivially ultrametric.
  At large δ: almost no nontrivially ultrametric triplets remain.
Recommended value: 0.1.
Allowed range: 0.01 – 0.50.
"""

# --- Computational Resource Parameters ---

NUM_WORKERS = None
"""
Number of parallel processes for generating neighbor structures.
  None: automatically use all available CPU cores.
  1: single-threaded mode (for debugging).
  N: use exactly N processes.
Recommended value: None.
Allowed range: 1 – cpu_count().
"""

VERBOSE = False
"""
Detailed output mode.
  True: print all intermediate results (basin sizes,
    transition statistics, triangle distribution).
  False: only final results (brief log).
Recommended value: True (for research purposes).
"""

# --- Statistical Analysis Parameter ---

NUM_STAT = 5
"""
Number of statistical trials (independent runs) for each
RNA sequence.
  NUM_STAT = 1: single run, result without deviation.
  NUM_STAT > 1: NUM_STAT runs are performed with different seeds
    (RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
    Results are averaged, displayed as mean ± SD.
    Integer quantities (number of structures, basins, components)
    are rounded to integers.
Recommended value: 1.
Allowed range: 1 – 100.
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

R_KCAL = 0.001987204259  # Gas constant in kcal/(mol·K) (R = N_A * k_B)
EPS_COMPARISON = 1e-9  # For comparing floating-point numbers



# ============================================================================
# OPTIMIZATION: BITMASKS AND PRECOMPUTATION OF ALLOWED PAIRS
# ============================================================================

def precompute_allowed_pairs(seq_len, sequence, min_hairpin_len, comp_map):
    """
    Precomputes a list of all allowed pairs (complementary and satisfying
    the minimum distance condition) and their corresponding bitmasks.
    
    Bitmasks allow checking pair conflicts in O(1) instead of O(k).
    A pair (i,j) is represented by a mask with bits i and j set.
    
    Arguments:
        seq_len (int): sequence length
        sequence (str): RNA sequence
        min_hairpin_len (int): minimum number of unpaired nucleotides in a loop
        comp_map (dict): complementarity dictionary
    
    Returns:
        list of tuples: [(i, j, mask), ...] — list of allowed pairs with their masks
    """
    allowed = []
    for i in range(seq_len):
        for j in range(i + min_hairpin_len + 1, seq_len):
            if (sequence[i], sequence[j]) in comp_map:
                mask = (1 << i) | (1 << j)
                allowed.append((i, j, mask))
    return allowed


def pairs_to_mask(pairs):
    """
    Converts a set of pairs into a bitmask of paired positions.
    
    Arguments:
        pairs (iterable of tuples): set of pairs (i,j)
    
    Returns:
        int: bitmask (bit k is set if position k is paired)
    """
    mask = 0
    for i, j in pairs:
        mask |= (1 << i) | (1 << j)
    return mask


# ============================================================================
# FUNCTIONS FOR WORKING WITH STRUCTURES
# ============================================================================

def dotbracket_to_pairs(structure):
    """
    Converts a dot-bracket structure into a set of base pairs.
    
    Arguments:
        structure (str): structure in dot-bracket notation (e.g., "(((...)))")
    
    Returns:
        frozenset of tuples: set of pairs (i, j), where i < j
    """
    pairs = []
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            pairs.append((j, i))
    return frozenset(pairs)


def pairs_to_dotbracket(pairs, length):
    """
    Converts a set of base pairs into a dot-bracket structure.
    
    Arguments:
        pairs (set of tuples): set of pairs (i, j), where i < j
        length (int): sequence length
    
    Returns:
        str: structure in dot-bracket notation
    """
    result = ['.'] * length
    for i, j in pairs:
        result[i] = '('
        result[j] = ')'
    return ''.join(result)


def deduplicate_structures(structures, energies, verbose=True):
    """
    Removes duplicate structures (with identical sets of pairs).
    If two structures have the same pairs, one is kept (with the minimum energy).
    
    Arguments:
        structures (list of str): list of structures in dot-bracket notation
        energies (np.ndarray): corresponding energies
        verbose (bool): whether to print a message about removed duplicates
    
    Returns:
        tuple: (new_structures, new_energies) after removing duplicates
    """
    unique_pairs = {}
    for s, e in zip(structures, energies):
        pairs = dotbracket_to_pairs(s)
        if pairs not in unique_pairs or e < unique_pairs[pairs][1]:
            unique_pairs[pairs] = (s, e)
    
    new_structures = []
    new_energies = []
    for s, e in unique_pairs.values():
        new_structures.append(s)
        new_energies.append(e)
    
    if verbose and len(new_structures) < len(structures):
        print(f"  Duplicates removed: {len(structures) - len(new_structures)}")
    
    return new_structures, np.array(new_energies)


def generate_neighbors_fast(pairs_fs, seq_len, allowed_pairs, pair_pos_mask_map):
    """
    OPTIMIZED version of neighbor generation using bitmasks.
    
    Neighborhood is defined via ONE operation:
    1. Removing one pair
    2. Adding one pair
    3. Shifting one pair (removal + addition)
    
    Uses a precomputed list of allowed pairs (allowed_pairs).
    Conflict checking via bitmasks — O(1).
    
    Arguments:
        pairs_fs (frozenset): set of pairs of the current structure
        seq_len (int): sequence length
        allowed_pairs (list of tuples): list (i, j, mask) of allowed pairs
        pair_pos_mask_map (dict): dictionary mapping (i,j) to its mask
    
    Returns:
        set of frozenset: set of neighboring structures (as sets of pairs)
    """
    neighbors = set()
    pairs_list = list(pairs_fs)
    # Cache the mask of the current structure
    current_mask = 0
    for i, j in pairs_list:
        current_mask |= (1 << i) | (1 << j)
    
    # Operation 1: Removing an existing pair
    for k in range(len(pairs_list)):
        new_pairs = set(pairs_list[:k] + pairs_list[k+1:])
        neighbors.add(frozenset(new_pairs))
    
    # Operation 2: Adding a new pair (only from the precomputed allowed list)
    for i, j, mask in allowed_pairs:
        # Check: positions i and j must not already be paired
        if current_mask & mask:
            continue
        # Check for absence of intersections with existing pairs
        has_conflict = False
        for a, b in pairs_list:
            if a < i < b < j or i < a < j < b:
                has_conflict = True
                break
        if not has_conflict:
            new_pairs = set(pairs_list)
            new_pairs.add((i, j))
            neighbors.add(frozenset(new_pairs))
    
    # Operation 3: Shifting a pair (removing an existing one + adding a new one)
    for idx_old, (i_old, j_old) in enumerate(pairs_list):
        # Mask without the pair being removed
        temp_mask = current_mask & ~((1 << i_old) | (1 << j_old))
        for i_new, j_new, mask_new in allowed_pairs:
            if i_new == i_old and j_new == j_old:
                continue
            # Check: new positions must not be occupied
            if temp_mask & mask_new:
                continue
            # Check for absence of intersections
            has_conflict = False
            for k in range(len(pairs_list)):
                if k == idx_old:
                    continue
                a, b = pairs_list[k]
                if a < i_new < b < j_new or i_new < a < j_new < b:
                    has_conflict = True
                    break
            if not has_conflict:
                new_pairs = set(pairs_list[:idx_old] + pairs_list[idx_old+1:])
                new_pairs.add((i_new, j_new))
                neighbors.add(frozenset(new_pairs))
    
    return neighbors


# ============================================================================
# OPTIMIZATION: PARALLEL NEIGHBOR GENERATION
# ============================================================================

def _pool_initializer(allowed_pairs, pair_pos_mask_map, seq_len):
    """
    Initializer for processes in the Pool.
    Sets global module variables so that each worker function
    can access them without passing through arguments (and without serialization).
    
    Arguments:
        allowed_pairs (list of tuples): list (i, j, mask) of allowed pairs
        pair_pos_mask_map (dict): dictionary (i,j) -> mask
        seq_len (int): sequence length
    """
    global _ALLOWED_PAIRS, _PAIR_POS_MASK_MAP, _SEQ_LEN
    _ALLOWED_PAIRS = allowed_pairs
    _PAIR_POS_MASK_MAP = pair_pos_mask_map
    _SEQ_LEN = seq_len


def _generate_neighbors_worker_indexed(args):
    """
    Worker function for parallel neighbor generation with index.
    Uses global variables _ALLOWED_PAIRS, _PAIR_POS_MASK_MAP, _SEQ_LEN.
    Takes a tuple (idx, pairs), returns a tuple (idx, neighbors).
    
    Arguments:
        args (tuple): (idx, pairs) — index of the structure and its set of pairs
    
    Returns:
        tuple: (idx, neighbors) — index of the structure and its set of neighbors
    """
    idx, pairs = args
    neighbors = generate_neighbors_fast(pairs, _SEQ_LEN, _ALLOWED_PAIRS, _PAIR_POS_MASK_MAP)
    return (idx, neighbors)


# ============================================================================
# GRAPH CONSTRUCTION AND ANALYSIS FUNCTIONS
# ============================================================================

def generate_structures_stochastic(seq, temp_celsius, max_structures, energy_window, verbose=True):
    """
    Generation of a set of secondary structures by stochastic sampling
    from the Boltzmann ensemble (pbacktrack) with an energy constraint.
    
    Arguments:
        seq (str): RNA sequence
        temp_celsius (float): temperature in °C
        max_structures (int): desired number of unique structures
        energy_window (float or str): energy window (kcal/mol) or "inf"
        verbose (bool): whether to print detailed messages
    
    Returns:
        tuple: (structures, energies)
            structures (list of str): structures in dot-bracket notation
            energies (np.ndarray): corresponding free energies
    """
    RNA.cvar.temperature = temp_celsius
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    
    # Step 1: MFE and partition function
    (mfe_struct, mfe) = fc.mfe()
    fc.pf()
    
    # Determine the energy threshold
    if isinstance(energy_window, str) and energy_window.lower() == "inf":
        energy_cutoff = float('inf')
    else:
        energy_cutoff = mfe + float(energy_window)
    
    # Step 2: Stochastic sampling
    structures = []
    energies_list = []
    seen = set()
    
    # First, add the MFE structure if it satisfies the energy criterion
    if mfe <= energy_cutoff + EPS_COMPARISON:
        structures.append(mfe_struct)
        energies_list.append(mfe)
        seen.add(mfe_struct)
    
    # Number of attempts with a margin
    batch_size = min(max_structures, 500)
    max_batches = (max_structures * 10) // batch_size + 1
    total_generated = 0
    total_rejected_energy = 0
    
    for batch in range(max_batches):
        if len(structures) >= max_structures:
            break
        try:
            # Generate a batch of random structures
            samples = fc.pbacktrack(batch_size)
            for struct in samples:
                if len(structures) >= max_structures:
                    break
                if not struct or len(struct) == 0:
                    continue
                if struct in seen:
                    continue
                total_generated += 1
                energy = fc.eval_structure(struct)
                # Check energy window
                if energy > energy_cutoff + EPS_COMPARISON:
                    total_rejected_energy += 1
                    continue
                seen.add(struct)
                structures.append(struct)
                energies_list.append(energy)
        except Exception:
            break
    
    if verbose:
        if total_rejected_energy > 0:
            print(f"  Structures discarded with energy above threshold ({energy_cutoff:.2f} kcal/mol): {total_rejected_energy}")
        print(f"  Stochastic sampling: generated {len(structures)} unique structures "
              f"(requested {max_structures})")
        if len(structures) < max_structures:
            print(f"  Warning: could not collect the requested number of structures within the specified energy window")
    
    return structures, np.array(energies_list)


def build_neighbor_graph(pair_sets, seq_len, allowed_pairs, pair_pos_mask_map, num_workers=None, verbose=True):
    """
    Builds a neighbor graph based on neighbor generation.
    
    Uses parallel neighbor generation via multiprocessing
    with global variables and stream processing.
    
    Arguments:
        pair_sets (list of frozenset): list of pair sets for all structures
        seq_len (int): sequence length
        allowed_pairs (list of tuples): precomputed list of allowed pairs
        pair_pos_mask_map (dict): dictionary (i,j) -> mask
        num_workers (int or None): number of processes (None = auto)
        verbose (bool): whether to print detailed messages
    
    Returns:
        list of set: for each structure — a set of neighbor indices
    """
    n_workers = num_workers if num_workers else cpu_count()
    n_structures = len(pair_sets)
    
    # Build an index for fast structure lookup by pair set
    index_map = {ps: i for i, ps in enumerate(pair_sets)}
    neighbors_list = [set() for _ in range(n_structures)]
    
    # Parallel neighbor generation
    if n_workers > 1:
        if verbose:
            print(f"  Using {n_workers} processes for neighbor generation")
            print(f"  Stream processing of results (imap_unordered) to save memory")
        indexed_args = [(idx, ps) for idx, ps in enumerate(pair_sets)]
        with Pool(n_workers, initializer=_pool_initializer,
                  initargs=(allowed_pairs, pair_pos_mask_map, seq_len)) as pool:
            for idx, neighbor_pairs in pool.imap_unordered(
                _generate_neighbors_worker_indexed, indexed_args, chunksize=10
            ):
                for nb_pairs in neighbor_pairs:
                    if nb_pairs in index_map:
                        neighbors_list[idx].add(index_map[nb_pairs])
                del neighbor_pairs
    else:
        for idx, ps in enumerate(pair_sets):
            neighbor_pairs = generate_neighbors_fast(ps, seq_len, allowed_pairs, pair_pos_mask_map)
            for nb_pairs in neighbor_pairs:
                if nb_pairs in index_map:
                    neighbors_list[idx].add(index_map[nb_pairs])
            del neighbor_pairs
    
    return neighbors_list


def find_connected_components(neighbors_list):
    """
    Finds connected components of the structure graph.
    Uses depth-first search.
    
    Arguments:
        neighbors_list (list of set): for each structure — a set of neighbor indices
    
    Returns:
        list of list: list of connected components, each component is a list of structure indices
    """
    n = len(neighbors_list)
    visited = [False] * n
    components = []
    
    for start in range(n):
        if not visited[start]:
            # Launch depth-first search from a new vertex
            component = []
            stack = [start]
            visited[start] = True
            while stack:
                v = stack.pop()
                component.append(v)
                for nb in neighbors_list[v]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(component)
    
    # Sort components by descending size
    components.sort(key=len, reverse=True)
    return components


def find_local_minima(energies, neighbors_list):
    """
    Finds all local minima in the neighbor graph.
    A structure is a local minimum if all its neighbors
    have STRICTLY GREATER energy.
    
    Arguments:
        energies (np.ndarray): array of energies
        neighbors_list (list of set): for each structure — a set of neighbor indices
    
    Returns:
        list of int: indices of local minima
    """
    local_minima = []
    for idx in range(len(energies)):
        is_min = True
        for nb in neighbors_list[idx]:
            if energies[nb] <= energies[idx] - EPS_COMPARISON:
                is_min = False
                break
        if is_min:
            local_minima.append(idx)
    return local_minima


def compute_gradient_basins(energies, neighbors_list, verbose=True):
    """
    Determines attraction basins (gradient basins)
    with correct handling of plateaus (regions of equal energy).
    
    Algorithm:
    1. Find all "attraction points": structures that have no neighbor
       with strictly lower energy. Among them, connected components
       with equal energies are identified — these are plateaus. Each such component
       forms one attraction point.
    2. For each structure, gradient descent is launched: at each step,
       the neighbor with the lowest energy is chosen (if equal — the smallest index
       for determinism). Descent continues until any structure
       belonging to one of the attraction points is reached.
    3. All structures that descended to the same attraction point form a basin
       of attraction (macrostate).
    
    Arguments:
        energies (np.ndarray): array of energies
        neighbors_list (list of set): for each structure — a set of neighbor indices
        verbose (bool): whether to print detailed messages
    
    Returns:
        list of tuples: [(min_idx, [indices]), ...] for each basin
    """
    n = len(energies)
    
    # ---- Step 1: Find all candidates for attraction points ----
    candidate_set = set()
    for i in range(n):
        has_lower = False
        for nb in neighbors_list[i]:
            if energies[nb] < energies[i] - EPS_COMPARISON:
                has_lower = True
                break
        if not has_lower:
            candidate_set.add(i)
    
    # ---- Step 2: Build plateau graph among candidates ----
    visited_candidate = set()
    attraction_points = []
    
    for v in candidate_set:
        if v not in visited_candidate:
            component = []
            stack = [v]
            visited_candidate.add(v)
            while stack:
                u = stack.pop()
                component.append(u)
                for nb in neighbors_list[u]:
                    if nb in candidate_set and nb not in visited_candidate:
                        if abs(energies[u] - energies[nb]) < EPS_COMPARISON:
                            visited_candidate.add(nb)
                            stack.append(nb)
            attraction_points.append(component)
    
    attraction_id = {}
    for idx, component in enumerate(attraction_points):
        for v in component:
            attraction_id[v] = idx
    
    if verbose:
        print(f"  Attraction points found: {len(attraction_points)}")
        plateau_count = sum(1 for comp in attraction_points if len(comp) > 1)
        if plateau_count > 0:
            plateau_sizes = [len(comp) for comp in attraction_points if len(comp) > 1]
            print(f"    Of these, plateaus (size > 1): {plateau_count}")
            print(f"    Plateau sizes: {plateau_sizes}")
    
    # ---- Step 3: Gradient descent ----
    basin_of = [-1] * n
    
    def find_basin(i):
        """Recursive descent from structure i to an attraction point."""
        if basin_of[i] != -1:
            return basin_of[i]
        
        if i in attraction_id:
            basin_of[i] = attraction_id[i]
            return attraction_id[i]
        
        neighbors = list(neighbors_list[i])
        
        if not neighbors:
            new_id = len(attraction_points)
            attraction_points.append([i])
            attraction_id[i] = new_id
            basin_of[i] = new_id
            return new_id
        
        best = min(neighbors, key=lambda x: (energies[x], x))
        
        if energies[best] >= energies[i] - EPS_COMPARISON:
            new_id = len(attraction_points)
            attraction_points.append([i])
            attraction_id[i] = new_id
            basin_of[i] = new_id
            return new_id
        
        basin = find_basin(best)
        basin_of[i] = basin
        return basin
    
    for i in range(n):
        find_basin(i)
    
    # ---- Step 4: Group structures by basins ----
    basins_dict = defaultdict(list)
    for idx, b in enumerate(basin_of):
        basins_dict[b].append(idx)
    
    basins = []
    for b, indices in basins_dict.items():
        representative = attraction_points[b][0]
        basins.append((representative, indices))
    
    basins.sort(key=lambda x: energies[x[0]])
    
    if verbose:
        print(f"  Number of macrostates (basins): {len(basins)}")
    
    return basins


# ============================================================================
# CONSTRUCTION OF THE TRANSITION RATE MATRIX K (STRUCTURE LEVEL)
# ============================================================================

def build_transition_rate_matrix(energies, neighbors_list, temp_kelvin, nu0):
    """
    Builds the symmetrized transition rate matrix K_sym between all
    structures based on the Kramers formula.
    
    For neighboring structures p and q:
        K_sym[p][q] = nu0 * exp(-|G(p) - G(q)| / (2 * R * T))
    
    Diagonal: K_sym[p][p] = -sum_{q != p} K_sym[p][q]
    
    Arguments:
        energies (np.ndarray): free energies of structures
        neighbors_list (list of set): list of neighbors for each structure
        temp_kelvin (float): temperature in Kelvin
        nu0 (float): frequency factor
    
    Returns:
        scipy.sparse.csr_matrix: sparse symmetric matrix K_sym (N x N)
    """
    N = len(energies)
    RT = R_KCAL * temp_kelvin
    
    # Use lil_matrix for construction, then convert to csr
    K_sym = lil_matrix((N, N), dtype=np.float64)
    
    # Temporary array for accumulating row sums
    row_sums = np.zeros(N, dtype=np.float64)
    
    for p in range(N):
        G_p = energies[p]
        
        for q in neighbors_list[p]:
            if q > p:  # Process each edge once
                G_q = energies[q]
                delta_G = abs(G_p - G_q)
                rate = nu0 * np.exp(-delta_G / (2.0 * RT))
                
                K_sym[p, q] = rate
                K_sym[q, p] = rate
                
                row_sums[p] += rate
                row_sums[q] += rate
    
    # Compute diagonal elements
    for p in range(N):
        K_sym[p, p] = -row_sums[p]
    
    return K_sym.tocsr()


def filter_eigenvalues_by_gap(eigenvalues, eigenvectors, gap_threshold):
    """
    Automatically finds the spectral gap and filters out noise modes.
    
    Algorithm:
    1. Sort eigenvalues by absolute value.
    2. Skip λ_0 ≈ 0 (stationary state).
    3. Search for a gap: if |λ_k| / |λ_{k-1}| > gap_threshold,
       then modes 0..k-1 are noise.
    4. Return filtered eigenvalues and eigenvectors.
    
    Arguments:
        eigenvalues (np.ndarray): array of eigenvalues
        eigenvectors (np.ndarray): matrix of eigenvectors (by columns)
        gap_threshold (float): threshold for detecting the gap
    
    Returns:
        tuple: (filtered_eigenvalues, filtered_eigenvectors, num_noise_modes)
    """
    # Sort by absolute value
    idx_sorted = np.argsort(np.abs(eigenvalues))
    sorted_vals = eigenvalues[idx_sorted]
    sorted_vecs = eigenvectors[:, idx_sorted]
    
    abs_vals = np.abs(sorted_vals)
    
    # Skip λ_0 (closest to zero — stationary state)
    # Search for a gap among the rest
    num_noise = 1  # at least λ_0 is the stationary state
    
    for k in range(1, len(abs_vals)):
        if abs_vals[k-1] < 1e-30:
            # Previous value is zero to machine precision
            ratio = float('inf') if abs_vals[k] > 1e-30 else 1.0
        else:
            ratio = abs_vals[k] / abs_vals[k-1]
        
        if ratio > gap_threshold:
            # Gap found between k-1 and k
            num_noise = k
            break
    else:
        # Gap not found — only λ_0 is considered noise
        num_noise = 1
    
    # Discard noise modes
    filtered_vals = sorted_vals[num_noise:]
    filtered_vecs = sorted_vecs[:, num_noise:]
    
    return filtered_vals, filtered_vecs, num_noise


def filter_macrostates_spectral(basins, Z, min_size, max_macrostates, verbose=True):
    """
    Filters macrostates by size and statistical significance.
    Returns filtered basins and a mapping from old indices to new ones.
    
    Arguments:
        basins (list of tuples): [(min_idx, [indices]), ...]
        Z (dict): {basin_index: statistical_sum}
        min_size (int): minimum basin size
        max_macrostates (int): maximum number of basins
        verbose (bool): whether to print detailed messages
    
    Returns:
        tuple: (filtered_basins, old_to_new_map)
    """
    valid = []
    for i, (_, indices) in enumerate(basins):
        if len(indices) >= min_size:
            valid.append(i)
    
    if verbose:
        print(f"  Macrostates excluded with size < {min_size}: {len(basins) - len(valid)}")
    
    if len(valid) > max_macrostates:
        valid.sort(key=lambda i: Z[i], reverse=True)
        valid = valid[:max_macrostates]
        if verbose:
            print(f"  Macrostates with the largest Z kept: {len(valid)} (out of {len(basins)})")
    else:
        valid.sort(key=lambda i: Z[i], reverse=True)
        if verbose:
            print(f"  Macrostates kept: {len(valid)}")
    
    filtered_basins = [basins[i] for i in valid]
    old_to_new = {old: new for new, old in enumerate(valid)}
    
    return filtered_basins, old_to_new


def compute_spectral_distance(
    K_sym,
    basins,
    num_modes_requested,
    temp_kelvin,
    gap_threshold,
    eigs_maxiter,
    eigs_sigma,
    verbose=True
):
    """
    Computes the Mahalanobis distance between basins via spectral
    decomposition of the symmetrized transition rate matrix K_sym.
    
    Algorithm:
    1. Partial diagonalization of K_sym: find num_modes_requested eigenmodes.
       If unsuccessful, use extended parameters and shift.
    2. Automatic filtering of noise modes via spectral gap detection.
    3. Construction of characteristic vectors of basins χ_A.
    4. Computation of the Mahalanobis distance.
    
    Arguments:
        K_sym (scipy.sparse.csr_matrix): symmetric K matrix (N x N)
        basins (list of tuples): [(min_idx, [indices]), ...] — basins
        num_modes_requested (int): requested number of eigenmodes
        temp_kelvin (float): temperature (for scaling the distance)
        gap_threshold (float): threshold for spectral gap detection
        eigs_maxiter (int): maximum number of iterations for ARPACK
        eigs_sigma (float): shift for searching eigenvalues near zero
        verbose (bool): whether to print detailed messages
    
    Returns:
        tuple: (dist_matrix, eigenvalues_used, eigenvectors_used, num_noise_modes)
    """
    N = K_sym.shape[0]
    K_basins = len(basins)
    
    # Check that num_modes_requested is valid
    max_possible = N - 1  # minus the zero mode
    if num_modes_requested > max_possible:
        if verbose:
            print(f"  Warning: requested modes ({num_modes_requested}) > N-1 ({max_possible})")
        num_modes_requested = max_possible
    
    # Determine ncv (number of Lanczos vectors) to improve convergence
    ncv = min(2 * num_modes_requested + 10, N)
    
    # --- Step 1: Partial diagonalization ---
    if verbose:
        print(f"  Computing {num_modes_requested} eigenmodes of matrix {N}x{N}...")
    
    eigenvalues = None
    eigenvectors = None
    last_error = None
    
    # Attempt 1: standard search for smallest magnitude
    try:
        eigenvalues, eigenvectors = eigsh(
            K_sym,
            k=num_modes_requested,
            which='SM',
            return_eigenvectors=True,
            maxiter=eigs_maxiter,
            ncv=ncv,
            tol=1e-8
        )
        if verbose:
            print(f"    Successful (method SM, {eigs_maxiter} iterations)")
    except Exception as e:
        last_error = e
        if verbose:
            print(f"    Method SM failed: {e}")
    
    # Attempt 2: search with shift (if the first attempt failed)
    if eigenvalues is None:
        try:
            eigenvalues, eigenvectors = eigsh(
                K_sym,
                k=num_modes_requested,
                which='LM',
                sigma=eigs_sigma,
                return_eigenvectors=True,
                maxiter=eigs_maxiter,
                ncv=ncv
            )
            if verbose:
                print(f"    Successful (method LM with shift sigma={eigs_sigma})")
        except Exception as e:
            last_error = e
            if verbose:
                print(f"    Method LM with shift failed: {e}")
    
    # If all attempts failed
    if eigenvalues is None:
        raise RuntimeError(
            f"Failed to compute eigenvalues after two attempts. "
            f"Last error: {last_error}"
        )
    
    # --- Step 2: Filter noise modes ---
    eigenvalues_filtered, eigenvectors_filtered, num_noise = filter_eigenvalues_by_gap(
        eigenvalues, eigenvectors, gap_threshold
    )
    
    num_phys = len(eigenvalues_filtered)
    
    if verbose:
        print(f"  Spectral analysis:")
        print(f"    Total modes found: {len(eigenvalues)}")
        print(f"    Noise modes discarded: {num_noise}")
        print(f"    Physical modes retained: {num_phys}")
        
        if num_noise > 1:
            abs_all = np.abs(np.sort(np.abs(eigenvalues)))
            print(f"    Noise modes (|λ|): {abs_all[:min(num_noise, 10)]}")
            if num_phys > 0:
                print(f"    First physical modes (|λ|): {np.abs(eigenvalues_filtered[:min(5, num_phys)])}")
                if num_noise > 0 and num_phys > 0:
                    gap_ratio = np.abs(eigenvalues_filtered[0]) / (abs_all[num_noise-1] + 1e-300)
                    print(f"    Gap ratio: {gap_ratio:.2e}")
    
    if num_phys == 0:
        raise RuntimeError(
            "All eigenmodes were filtered out as noise. "
            "Check SPECTRAL_GAP_THRESHOLD parameter and temperature."
        )
    
    # --- Step 3: Characteristic vectors of basins ---
    if verbose:
        print(f"  Constructing characteristic vectors for {K_basins} basins...")
    
    chi = np.zeros((K_basins, N), dtype=np.float64)
    for a, (_, indices) in enumerate(basins):
        norm = np.sqrt(len(indices))
        chi[a, indices] = 1.0 / norm
    
    # --- Step 4: Projections onto eigenvectors ---
    proj = chi @ eigenvectors_filtered  # (K_basins x num_phys)
    
    # Weights: 1/|λ_k|
    weights = 1.0 / np.abs(eigenvalues_filtered)  # (num_phys,)
    
    # --- Step 5: Compute distances ---
    if verbose:
        print(f"  Computing distance matrix {K_basins}x{K_basins}...")
    
    dist_matrix = np.zeros((K_basins, K_basins), dtype=np.float64)
    
    for a in range(K_basins):
        for b in range(a + 1, K_basins):
            diff = proj[a, :] - proj[b, :]  # (num_phys,)
            d_sq = np.sum(weights * (diff ** 2))
            dist_matrix[a, b] = np.sqrt(d_sq)
            dist_matrix[b, a] = dist_matrix[a, b]
    
    # Physical normalization: multiply by RT to bring to energy units
    dist_matrix *= R_KCAL * temp_kelvin
    
    return dist_matrix, eigenvalues_filtered, eigenvectors_filtered, num_noise


# ============================================================================
# ULTRAMETRICITY CHECK
# ============================================================================

def classify_triangle(d1, d2, d3, eps, delta):
    """
    Classification of a triangle by ultrametricity.
    Uses branching instead of sorted() for speed.
    
    Arguments:
        d1, d2, d3 (float): sides of the triangle
        eps (float): accuracy for equality of the two largest sides
        delta (float): minimum difference for nontrivial ultrametricity
    
    Returns:
        str: 'trivial', 'nontrivial', or 'non_ultrametric'
    """
    if d1 == float('inf') or d2 == float('inf') or d3 == float('inf'):
        return 'non_ultrametric'
    
    # Determine min, mid, max without sorted()
    if d1 <= d2:
        if d1 <= d3:
            d_min = d1
            if d2 <= d3:
                d_mid, d_max = d2, d3
            else:
                d_mid, d_max = d3, d2
        else:
            d_min, d_mid, d_max = d3, d1, d2
    else:
        if d2 <= d3:
            d_min = d2
            if d1 <= d3:
                d_mid, d_max = d1, d3
            else:
                d_mid, d_max = d3, d1
        else:
            d_min, d_mid, d_max = d3, d2, d1
    
    if d_max <= EPS_COMPARISON:
        return 'trivial'
    
    if d_mid > EPS_COMPARISON:
        cond1 = (d_max - d_mid) / d_mid <= eps
        cond2 = (d_mid - d_min) / d_mid > delta
        if cond1 and cond2:
            return 'nontrivial'
    
    if d_min > EPS_COMPARISON:
        if (d_max - d_min) / d_min <= eps:
            return 'trivial'
    elif d_max <= EPS_COMPARISON:
        return 'trivial'
    
    return 'non_ultrametric'


def compute_ultrametricity_score(dist_matrix, eps, delta):
    """Computation of ultrametricity degrees (nontrivial, trivial, non-ultrametric)."""
    n = dist_matrix.shape[0]
    if n < 3:
        return 0.0, 0.0, 0.0, defaultdict(int)
    
    triplets = list(combinations(range(n), 3))
    if not triplets:
        return 0.0, 0.0, 0.0, defaultdict(int)
    
    counts = defaultdict(int)
    
    for i, j, k in triplets:
        cls = classify_triangle(
            dist_matrix[i, j],
            dist_matrix[i, k],
            dist_matrix[j, k],
            eps, delta
        )
        counts[cls] += 1
    
    total = len(triplets)
    u_nt = counts.get('nontrivial', 0) / total * 100
    u_tr = counts.get('trivial', 0) / total * 100
    u_non = counts.get('non_ultrametric', 0) / total * 100
    
    return u_nt, u_tr, u_non, counts


# ============================================================================
# FUNCTION FOR LOADING SEQUENCES FROM FASTA FILES
# ============================================================================

def load_fasta_sequences():
    """
    Scans the current folder for files with the .fasta extension,
    reads RNA sequences using Biopython, sorts them
    by increasing length, and returns a list of tuples
    (sequence, description_full, length).
    """
    try:
        from Bio import SeqIO
    except ImportError:
        print("Error: the biopython package is required to work with FASTA files.")
        print("Install it with: pip install biopython")
        raise

    fasta_files = glob.glob("*.fasta")
    
    if not fasta_files:
        print("Error: no files with .fasta extension found in the current folder")
        return []
    
    print(f"FASTA files found: {len(fasta_files)}")
    for f in fasta_files:
        print(f"  {f}")
    
    all_sequences = []
    for fasta_file in fasta_files:
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                seq_str = str(record.seq).upper()
                desc_full = record.description if record.description else record.id
                valid_chars = {'A', 'U', 'G', 'C'}
                seq_str = seq_str.replace('T', 'U')
                filtered_seq = ''.join(c for c in seq_str if c in valid_chars)
                if len(filtered_seq) < len(seq_str):
                    print(f"  Warning: invalid characters found in sequence {desc_full}, they were skipped.")
                if len(filtered_seq) < 10:
                    print(f"  Warning: sequence {desc_full} is too short (< 10 nt), skipping.")
                    continue
                all_sequences.append((filtered_seq, desc_full, len(filtered_seq)))
        except Exception as e:
            print(f"  Error reading file {fasta_file}: {e}")
    
    if not all_sequences:
        print("Error: failed to load any sequences from FASTA files")
        return []
    
    all_sequences.sort(key=lambda x: x[2])
    
    print(f"\nSequences loaded: {len(all_sequences)}")
    print("Sequences (in order of increasing length):")
    for i, (seq, desc, length) in enumerate(all_sequences):
        print(f"  {i+1}. {desc}: length {length} nt")
    
    return all_sequences


# ============================================================================
# FUNCTION FOR PROCESSING A SINGLE SEQUENCE (SINGLE RUN)
# ============================================================================

def process_sequence_single(seq, seq_description, seq_index, total_sequences, 
                            stat_iter, num_stat, current_seed, show_details=True):
    """
    Performs a complete ultrametricity analysis for a single RNA sequence
    using the spectral Mahalanobis distance.
    SINGLE RUN (used both for NUM_STAT=1 and in the loop when NUM_STAT>1).
    
    Arguments:
        seq (str): RNA sequence (A, U, G, C)
        seq_description (str): sequence description
        seq_index (int): ordinal number of the sequence
        total_sequences (int): total number of sequences
        stat_iter (int): current iteration number (starting from 1), for output
        num_stat (int): total number of statistical trials
        current_seed (int): seed for this iteration
        show_details (bool): whether to print detailed messages (steps, components, etc.)
    
    Returns:
        dict: dictionary with analysis results or None.
    """
    start_time = time.time()
    step_timings = {}
    
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15
    RT = R_KCAL * temp_kelvin
    
    np.random.seed(current_seed)
    
    # ===== STEP 1: PRECOMPUTATION OF ALLOWED PAIRS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 1: PRECOMPUTATION OF ALLOWED PAIRS")
        print("-" * 50)
    
    comp_map = {
        ('A', 'U'): True, ('U', 'A'): True,
        ('G', 'C'): True, ('C', 'G'): True,
        ('G', 'U'): True, ('U', 'G'): True,
    }
    allowed_pairs = precompute_allowed_pairs(seq_len, seq, MIN_HAIRPIN_LEN, comp_map)
    pair_pos_mask_map = {(i, j): mask for i, j, mask in allowed_pairs}
    if show_details:
        print(f"  Total allowed pairs: {len(allowed_pairs)}")
    step_timings['Step 1: Precomputation of allowed pairs'] = time.time() - step_start
    
    # ===== STEP 2: STRUCTURE GENERATION =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 2: GENERATION OF SECONDARY STRUCTURES (stochastic sampling, pbacktrack)")
        print("-" * 50)
    
    structures, energies = generate_structures_stochastic(
        seq, TEMPERATURE_CELSIUS, MAX_STRUCTURES, ENERGY_WINDOW, verbose=show_details
    )
    
    if show_details:
        print(f"Structures generated: {len(structures)}")
    step_timings['Step 2: Structure generation'] = time.time() - step_start
    
    if len(structures) < 2:
        print("Error: insufficient structures for analysis")
        gc.collect()
        return None
    
    # ===== STEP 3: REMOVAL OF DUPLICATES =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 3: REMOVAL OF DUPLICATES")
        print("-" * 50)
    
    structures, energies = deduplicate_structures(structures, energies, verbose=show_details)
    if show_details:
        print(f"Unique structures: {len(structures)}")
    step_timings['Step 3: Removal of duplicates'] = time.time() - step_start
    
    # ===== STEP 4: CONVERSION TO PAIRS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 4: CONVERSION TO PAIR SETS")
        print("-" * 50)
    
    pair_sets = [dotbracket_to_pairs(s) for s in structures]
    step_timings['Step 4: Conversion to pairs'] = time.time() - step_start
    
    # ===== STEP 5: BUILDING THE NEIGHBOR GRAPH =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 5: BUILDING THE NEIGHBOR GRAPH")
        print("-" * 50)
        print("  Parallel generation with global variables and stream processing")
    
    neighbors = build_neighbor_graph(pair_sets, seq_len, allowed_pairs, pair_pos_mask_map, n_workers, verbose=show_details)
    edges = sum(len(nb) for nb in neighbors) // 2
    if show_details:
        print(f"  Graph built: {len(neighbors)} vertices, {edges} edges")
    step_timings['Step 5: Building the neighbor graph'] = time.time() - step_start
    
    del pair_sets
    gc.collect()
    
    # ===== STEP 6: CHECKING GRAPH CONNECTIVITY =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 6: CHECKING GRAPH CONNECTIVITY")
        print("-" * 50)
    
    graph_components = find_connected_components(neighbors)
    num_components = len(graph_components)
    component_sizes = [len(comp) for comp in graph_components]
    if show_details:
        print(f"  Connected components found: {num_components}")
        print(f"  Component sizes (first 15): {component_sizes[:15]}")
    step_timings['Step 6: Checking graph connectivity'] = time.time() - step_start
    
    # ===== STEP 7: FINDING LOCAL MINIMA (FOR THE WHOLE GRAPH) =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 7: FINDING LOCAL MINIMA")
        print("-" * 50)
    
    local_minima = find_local_minima(energies, neighbors)
    if show_details:
        print(f"Local minima found: {len(local_minima)}")
    step_timings['Step 7: Finding local minima'] = time.time() - step_start
    
    # ===== STEP 8: DETERMINING MACROSTATES =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 8: DETERMINING MACROSTATES")
        print("-" * 50)
    
    basins = compute_gradient_basins(energies, neighbors, verbose=show_details)
    total_basins = len(basins)
    if show_details:
        print(f"Number of macrostates (before filtering): {total_basins}")
    step_timings['Step 8: Determining macrostates'] = time.time() - step_start
    
    # ===== STEP 9: COMPUTING STATISTICAL SUMS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 9: COMPUTING STATISTICAL SUMS OF BASINS")
        print("-" * 50)
    
    Z = {}
    for i, (_, indices) in enumerate(basins):
        Z[i] = sum(np.exp(-energies[idx] / RT) for idx in indices)
    
    if show_details:
        print(f"  Statistical sums computed for {total_basins} basins")
    step_timings['Step 9: Computing statsums'] = time.time() - step_start
    
    # ===== STEP 10: FILTERING MACROSTATES (GLOBAL) =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 10: FILTERING MACROSTATES")
        print("-" * 50)
    
    filtered_basins, old_to_new = filter_macrostates_spectral(
        basins, Z, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, verbose=show_details
    )
    num_filtered_basins = len(filtered_basins)
    if show_details:
        print(f"Number of macrostates (after filtering): {num_filtered_basins}")
    step_timings['Step 10: Filtering macrostates'] = time.time() - step_start
    
    if num_filtered_basins < 3:
        print("Error: fewer than 3 macrostates remain after filtering")
        gc.collect()
        return None
    
    # ===== STEP 11: PROCESSING BY CONNECTED COMPONENTS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 11: PROCESSING BY CONNECTED COMPONENTS")
        print("-" * 50)
    
    # If the graph is connected — process as a single component
    if num_components == 1:
        if show_details:
            print(f"  Graph is connected. Processing as a single component.")
        components_to_process = [(0, list(range(len(energies))))]
    else:
        if show_details:
            print(f"  Graph is disconnected ({num_components} components). Processing each component separately.")
        # Filter components: keep only those containing at least 3 basins
        components_to_process = []
        skipped_by_size = defaultdict(int)  # size -> number of skipped components
        
        for comp_idx, comp_indices in enumerate(graph_components):
            comp_set = set(comp_indices)
            # Count basins that have at least one structure in this component
            basins_in_comp = 0
            for basin_idx, (_, basin_structs) in enumerate(filtered_basins):
                if any(s in comp_set for s in basin_structs):
                    basins_in_comp += 1
            if basins_in_comp >= 3:
                components_to_process.append((comp_idx, comp_indices))
                if show_details:
                    print(f"    Component {comp_idx}: size {len(comp_indices)}, basins ~ {basins_in_comp}")
            else:
                skipped_by_size[len(comp_indices)] += 1
        
        # Print summary of skipped components
        if show_details and skipped_by_size:
            print(f"    Components skipped (basins < 3):")
            for size in sorted(skipped_by_size.keys(), reverse=True):
                count = skipped_by_size[size]
                print(f"      size {size}: {count} component{'s' if count != 1 else ''}")
    
    if not components_to_process:
        print("  Error: no connected component contains >= 3 basins")
        gc.collect()
        return None
    
    # ===== STEP 12: BUILDING K_sym AND SPECTRAL ANALYSIS FOR EACH COMPONENT =====
    all_component_results = []
    global_basin_stats = []  # for the final report
    
    for comp_idx, comp_indices in components_to_process:
        if show_details:
            print(f"\n  --- Component {comp_idx} ({len(comp_indices)} structures) ---")
        
        # 12.1: Extract component structures
        comp_set = set(comp_indices)
        comp_energies = energies[list(comp_indices)]
        
        # Build mapping from old indices to new ones (local to the component)
        old_to_local = {old: local for local, old in enumerate(comp_indices)}
        
        # 12.2: Extract basins belonging to this component
        comp_basins = []
        for basin_idx, (rep, basin_structs) in enumerate(filtered_basins):
            # Structures of the basin that fell into this component
            local_structs = [old_to_local[s] for s in basin_structs if s in comp_set]
            if len(local_structs) > 0:
                # Representative — the first structure of the basin in this component
                local_rep = local_structs[0]
                comp_basins.append((local_rep, local_structs))
        
        if len(comp_basins) < 3:
            if show_details:
                print(f"    Skipped: only {len(comp_basins)} basins (< 3)")
            continue
        
        if show_details:
            print(f"    Basins in component: {len(comp_basins)}")
        
        # 12.3: Build neighbor graph for the component
        comp_neighbors = []
        for old_idx in comp_indices:
            local_idx = old_to_local[old_idx]
            local_nbs = set()
            for nb in neighbors[old_idx]:
                if nb in comp_set:
                    local_nbs.add(old_to_local[nb])
            comp_neighbors.append(local_nbs)
        
        # 12.4: Build K_sym for the component
        if show_details:
            print(f"    Building K_sym ({len(comp_indices)} x {len(comp_indices)})...")
        K_sym_comp = build_transition_rate_matrix(
            comp_energies, comp_neighbors, temp_kelvin, FREQUENCY_PREFACTOR
        )
        if show_details:
            print(f"    Non-zero elements: {K_sym_comp.nnz}")
        
        # 12.5: Spectral decomposition and distance computation
        max_possible = len(comp_indices) - 1
        num_requested = min(NUM_EIGENMODES, max_possible)
        
        try:
            dist_matrix_comp, eigenvalues_used, eigenvectors_used, num_noise = compute_spectral_distance(
                K_sym_comp,
                comp_basins,
                num_requested,
                temp_kelvin,
                SPECTRAL_GAP_THRESHOLD,
                EIGS_MAXITER,
                EIGS_SIGMA,
                verbose=show_details
            )
        except Exception as e:
            print(f"    ERROR during spectral decomposition: {e}")
            print(f"    Component skipped.")
            continue
        
        # 12.6: Ultrametricity check
        u_nt_comp, u_tr_comp, u_non_comp, counts_comp = compute_ultrametricity_score(
            dist_matrix_comp, ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA
        )
        
        if show_details:
            print(f"    Nontrivial ultrametricity degree: {u_nt_comp:.2f}%")
            if counts_comp:
                total_triplets = sum(counts_comp.values())
                for cls, cnt in sorted(counts_comp.items()):
                    pct = cnt / total_triplets * 100 if total_triplets > 0 else 0
                    print(f"      {cls}: {cnt} ({pct:.1f}%)")
        
        all_component_results.append({
            'comp_idx': comp_idx,
            'comp_size': len(comp_indices),
            'num_basins': len(comp_basins),
            'num_phys_modes': len(eigenvalues_used),
            'num_noise_modes': num_noise,
            'u_nt': u_nt_comp,
            'u_tr': u_tr_comp,
            'u_non': u_non_comp,
            'counts': dict(counts_comp),
            'dist_matrix': dist_matrix_comp,
            'basins': comp_basins
        })
        
        global_basin_stats.append({
            'comp_idx': comp_idx,
            'comp_size': len(comp_indices),
            'num_basins': len(comp_basins),
            'u_nt': u_nt_comp,
            'u_tr': u_tr_comp,
            'u_non': u_non_comp,
            'num_triplets': sum(counts_comp.values())
        })
    
    step_timings['Step 11-12: Component processing and spectral analysis'] = time.time() - step_start
    
    if not all_component_results:
        print("\n  Error: failed to process any connected component")
        gc.collect()
        return None
    
    # ===== COMPUTING WEIGHTED AVERAGE ULTRAMETRICITY DEGREES =====
    total_triplets_all = sum(r['num_triplets'] for r in global_basin_stats)
    if total_triplets_all > 0:
        weighted_u_nt = sum(r['u_nt'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
        weighted_u_tr = sum(r['u_tr'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
        weighted_u_non = sum(r['u_non'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
    else:
        weighted_u_nt = 0.0
        weighted_u_tr = 0.0
        weighted_u_non = 0.0
    
    # ===== OUTPUT OF RESULTS (only in detailed mode) =====
    if show_details:
        print("\n" + "-" * 50)
        print("FINAL RESULTS")
        print("-" * 50)
        print(f"  Description: {seq_description}")
        print(f"  Length: {seq_len} nucleotides")
        print(f"  Method: spectral Mahalanobis distance")
        print(f"  Number of structures: {len(energies)}")
        print(f"  Number of connected components: {num_components}")
        print(f"  Number of processed components: {len(all_component_results)}")
        print(f"  Number of basins (total after filtering): {num_filtered_basins}")
        print(f"  Weighted nontrivial ultrametricity degree: {weighted_u_nt:.2f}%")
        print(f"  Weighted trivial ultrametricity degree: {weighted_u_tr:.2f}%")
        print(f"  Weighted non-ultrametricity degree: {weighted_u_non:.2f}%")
        
        if len(all_component_results) > 1:
            print(f"\n  Results by component:")
            for res in all_component_results:
                print(f"    Component {res['comp_idx']}: size {res['comp_size']}, "
                      f"basins {res['num_basins']}, "
                      f"phys. modes {res['num_phys_modes']}, "
                      f"noise {res['num_noise_modes']}, "
                      f"u_nt = {res['u_nt']:.2f}%")
        
        # Timings
        print("\n" + "-" * 50)
        print("STEP TIMINGS")
        print("-" * 50)
        for step_name, step_elapsed in step_timings.items():
            if step_elapsed < 60:
                print(f"  {step_name}: {step_elapsed:.1f} sec")
            else:
                minutes = int(step_elapsed // 60)
                seconds = step_elapsed % 60
                print(f"  {step_name}: {minutes} min {seconds:.1f} sec")
    
    elapsed = time.time() - start_time
    if show_details:
        print(f"  {'—' * 40}")
        if elapsed < 60:
            print(f"  TOTAL TIME: {elapsed:.1f} sec")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            print(f"  TOTAL TIME: {minutes} min {seconds:.1f} sec")
    
    gc.collect()
    
    # Use the results of the first (or only) component for the summary report
    first_res = all_component_results[0]
    
    return {
        'sequence': seq[:50],
        'description': seq_description,
        'length': seq_len,
        'weighted_u_nt': weighted_u_nt,
        'weighted_u_tr': weighted_u_tr,
        'weighted_u_non': weighted_u_non,
        'n_components': num_components,
        'n_processed_components': len(all_component_results),
        'n_structures': len(energies),
        'n_basins': num_filtered_basins,
        'component_results': all_component_results,
        'counts': first_res['counts'],
        'method': 'spectral_mahalanobis',
        'elapsed': elapsed,
        'step_timings': step_timings
    }


# ============================================================================
# FUNCTION FOR PROCESSING A SINGLE SEQUENCE (TAKING INTO ACCOUNT NUM_STAT)
# ============================================================================

def process_sequence(seq, seq_description, seq_index, total_sequences):
    """
    Performs a complete ultrametricity analysis for a single RNA sequence.
    When NUM_STAT > 1, performs several runs and averages the results.
    
    Arguments:
        seq (str): RNA sequence (A, U, G, C)
        seq_description (str): sequence description
        seq_index (int): ordinal number
        total_sequences (int): total number of sequences
    
    Returns:
        dict: dictionary with averaged analysis results or None.
    """
    all_runs = []
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15
    RT = R_KCAL * temp_kelvin
    
    # Print sequence header (once)
    if NUM_STAT > 1:
        header = f"PROCESSING SEQUENCE {seq_index} OF {total_sequences}"
    else:
        header = f"PROCESSING SEQUENCE {seq_index} OF {total_sequences}"
    
    print("\n" + "=" * 70)
    print(header)
    print(f"Description: {seq_description}")
    print(f"Length: {seq_len} nucleotides")
    first_seed = RANDOM_SEED
    print(f"Seed: {first_seed}")
    print("=" * 70)
    
    display_seq = seq if len(seq) <= 200 else seq[:197] + "..."
    wrapped_seq = "\n    ".join(textwrap.wrap(display_seq, width=60))
    print(f"\nSequence:\n    {wrapped_seq}")
    if len(seq) > 200:
        print(f"    (showing first 200 of {len(seq)} nucleotides)")
    
    # Print parameters (once)
    print("\n" + "-" * 50)
    print("SIMULATION PARAMETERS")
    print("-" * 50)
    print(f"Temperature: {TEMPERATURE_CELSIUS:.2f}°C ({TEMPERATURE_CELSIUS + 273.15:.2f} K)")
    if isinstance(ENERGY_WINDOW, str) and ENERGY_WINDOW.lower() == "inf":
        print(f"Energy window: none (inf)")
    else:
        print(f"Energy window: {ENERGY_WINDOW} kcal/mol")
    print(f"Max structures: {MAX_STRUCTURES}")
    print(f"Min macrostate size: {MIN_MACROSTATE_SIZE}")
    print(f"Max number of macrostates: {MAX_MACROSTATES_ANALYSIS}")
    print(f"Number of requested eigenmodes: {NUM_EIGENMODES}")
    print(f"Spectral gap threshold: {SPECTRAL_GAP_THRESHOLD:.1e}")
    print(f"Frequency factor ν₀: {FREQUENCY_PREFACTOR}")
    print(f"Max ARPACK iterations: {EIGS_MAXITER}")
    print(f"ARPACK sigma shift: {EIGS_SIGMA}")
    print(f"Min hairpin length: {MIN_HAIRPIN_LEN} (j - i >= {MIN_HAIRPIN_LEN + 1})")
    print(f"Accuracy ε: {ULTRAMETRIC_EPSILON}, δ: {ULTRAMETRIC_DELTA}")
    print(f"Seed: {RANDOM_SEED} (base)")
    if NUM_STAT > 1:
        print(f"NUM_STAT: {NUM_STAT} (statistical trials)")
    print(f"Number of processes (CPU): {n_workers}")
    print(f"Method: spectral Mahalanobis distance with auto noise filtering")
    print(f"\nR·T = {RT:.6f} kcal/mol")
    
    # Loop of runs
    for run_idx in range(NUM_STAT):
        current_seed = RANDOM_SEED + run_idx
        
        if not VERBOSE and NUM_STAT > 1:
            print(f"\nRUN {run_idx + 1}/{NUM_STAT}")
        
        # In quiet mode, suppress details inside process_sequence_single
        show_details = VERBOSE
        
        result = process_sequence_single(
            seq, seq_description, seq_index, total_sequences,
            stat_iter=run_idx + 1, num_stat=NUM_STAT, current_seed=current_seed,
            show_details=show_details
        )
        if result is not None:
            all_runs.append(result)
        
        if not VERBOSE and NUM_STAT > 1:
            print(f"COMPLETED {run_idx + 1}/{NUM_STAT}")
        
        gc.collect()
    
    if not all_runs:
        print(f"\nError: all runs for sequence {seq_index} failed")
        return None
    
    if NUM_STAT == 1:
        # Single run — print the result and return as is
        res = all_runs[0]
        print("\n" + "=" * 70)
        print(f"RESULT FOR SEQUENCE {seq_index}")
        print("=" * 70)
        print(f"  Weighted u_nt:        {res['weighted_u_nt']:.2f} %")
        print(f"  Weighted u_tr:        {res['weighted_u_tr']:.2f} %")
        print(f"  Weighted u_non:       {res['weighted_u_non']:.2f} %")
        print(f"  Number of structures:               {res['n_structures']}")
        print(f"  Number of basins:              {res['n_basins']}")
        print(f"  Number of connected components:    {res['n_components']}")
        print(f"  Execution time:             {res['elapsed']:.1f} sec")
        return res
    
    # Multiple runs — compute averages and SD
    print("\n" + "=" * 70)
    print(f"STATISTICS OVER {len(all_runs)} RUNS FOR SEQUENCE {seq_index}")
    print("=" * 70)
    
    # Extract arrays of values
    weighted_u_nt_vals = np.array([r['weighted_u_nt'] for r in all_runs])
    weighted_u_tr_vals = np.array([r['weighted_u_tr'] for r in all_runs])
    weighted_u_non_vals = np.array([r['weighted_u_non'] for r in all_runs])
    n_structures_vals = np.array([r['n_structures'] for r in all_runs], dtype=np.float64)
    n_basins_vals = np.array([r['n_basins'] for r in all_runs], dtype=np.float64)
    n_components_vals = np.array([r['n_components'] for r in all_runs], dtype=np.float64)
    elapsed_vals = np.array([r['elapsed'] for r in all_runs])
    
    # Averages and SD
    mean_u_nt = np.mean(weighted_u_nt_vals)
    std_u_nt = np.std(weighted_u_nt_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_u_tr = np.mean(weighted_u_tr_vals)
    std_u_tr = np.std(weighted_u_tr_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_u_non = np.mean(weighted_u_non_vals)
    std_u_non = np.std(weighted_u_non_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_struct = np.mean(n_structures_vals)
    std_struct = np.std(n_structures_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_basins = np.mean(n_basins_vals)
    std_basins = np.std(n_basins_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_comp = np.mean(n_components_vals)
    std_comp = np.std(n_components_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_elapsed = np.mean(elapsed_vals)
    std_elapsed = np.std(elapsed_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    # Round integer quantities to integers
    mean_struct_rounded = int(round(mean_struct))
    std_struct_rounded = int(round(std_struct))
    mean_basins_rounded = int(round(mean_basins))
    std_basins_rounded = int(round(std_basins))
    mean_comp_rounded = int(round(mean_comp))
    std_comp_rounded = int(round(std_comp))
    
    print(f"  Weighted u_nt:        {mean_u_nt:.2f} ± {std_u_nt:.2f} %")
    print(f"  Weighted u_tr:        {mean_u_tr:.2f} ± {std_u_tr:.2f} %")
    print(f"  Weighted u_non:       {mean_u_non:.2f} ± {std_u_non:.2f} %")
    print(f"  Number of structures:               {mean_struct_rounded} ± {std_struct_rounded}")
    print(f"  Number of basins:              {mean_basins_rounded} ± {std_basins_rounded}")
    print(f"  Number of connected components:    {mean_comp_rounded} ± {std_comp_rounded}")
    print(f"  Execution time:             {mean_elapsed:.1f} ± {std_elapsed:.1f} sec")
    
    # Return aggregated result
    first_res = all_runs[0]
    return {
        'sequence': first_res['sequence'],
        'description': first_res['description'],
        'length': first_res['length'],
        'weighted_u_nt': mean_u_nt,
        'weighted_u_nt_std': std_u_nt,
        'weighted_u_tr': mean_u_tr,
        'weighted_u_tr_std': std_u_tr,
        'weighted_u_non': mean_u_non,
        'weighted_u_non_std': std_u_non,
        'n_structures': mean_struct_rounded,
        'n_structures_std': std_struct_rounded,
        'n_basins': mean_basins_rounded,
        'n_basins_std': std_basins_rounded,
        'n_components': mean_comp_rounded,
        'n_components_std': std_comp_rounded,
        'n_processed_components': first_res['n_processed_components'],
        'component_results': first_res['component_results'],
        'counts': first_res['counts'],
        'method': first_res['method'],
        'elapsed': mean_elapsed,
        'elapsed_std': std_elapsed,
        'num_runs': len(all_runs),
        'all_runs': all_runs  # save all runs for detailed analysis
    }


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    total_start_time = time.time()
    
    print("=" * 70)
    print("CALCULATION OF THE DEGREE OF NONTRIVIAL ULTRAMETRICITY")
    print("FOR MACROSTATES OF RNA SECONDARY STRUCTURE")
    print("METHOD: spectral Mahalanobis distance")
    print("(physically rigorous approach via the transition rate matrix)")
    print("STRUCTURE GENERATION: stochastic sampling (pbacktrack)")
    if NUM_STAT > 1:
        print(f"STATISTICAL MODE: {NUM_STAT} runs per sequence")
    print("=" * 70)
    
    if FASTA_RNA:
        sequences = load_fasta_sequences()
        if not sequences:
            return
    else:
        sequences = [(RNA_SEQUENCE, "RNA_SEQUENCE (from parameter)", len(RNA_SEQUENCE))]
    
    all_seq_results = []
    
    for seq_idx, (seq, seq_desc, seq_len) in enumerate(sequences, start=1):
        result = process_sequence(seq, seq_desc, seq_idx, len(sequences))
        if result is not None:
            all_seq_results.append(result)
        gc.collect()
    
    # Final summary report
    if len(sequences) > 1 and all_seq_results:
        print("\n" + "=" * 70)
        print("FINAL SUMMARY REPORT")
        print("=" * 70)
        
        if NUM_STAT > 1:
            # Extended header with ± SD
            header = (f"{'№':<4} {'Description':<30} {'Length':<8} "
                      f"{'Structures':<16} {'Comp':<14} {'Basins':<16} "
                      f"{'u_nt (%)':<18} {'u_tr (%)':<18} {'u_non (%)':<18} {'Time (s)':<16}")
        else:
            header = (f"{'№':<4} {'Description':<35} {'Length':<8} {'Structures':<10} "
                      f"{'Comp':<6} {'Basins':<12} "
                      f"{'u_nt (%)':<12} {'u_tr (%)':<12} {'u_non (%)':<12} {'Time (s)':<10}")
        print(header)
        print("-" * len(header))
        
        for i, res in enumerate(all_seq_results):
            desc = res['description'][:28] if NUM_STAT > 1 else res['description'][:33]
            
            if NUM_STAT > 1:
                struct_str = f"{int(res['n_structures'])}±{int(res['n_structures_std'])}"
                comp_str = f"{int(res['n_components'])}±{int(res['n_components_std'])}"
                basins_str = f"{int(res['n_basins'])}±{int(res['n_basins_std'])}"
                u_nt_str = f"{res['weighted_u_nt']:.2f}±{res['weighted_u_nt_std']:.2f}"
                u_tr_str = f"{res['weighted_u_tr']:.2f}±{res['weighted_u_tr_std']:.2f}"
                u_non_str = f"{res['weighted_u_non']:.2f}±{res['weighted_u_non_std']:.2f}"
                time_str = f"{res['elapsed']:.1f}±{res['elapsed_std']:.1f}"
                print(f"{i+1:<4} {desc:<30} {res['length']:<8} "
                      f"{struct_str:<16} {comp_str:<14} {basins_str:<16} "
                      f"{u_nt_str:<18} {u_tr_str:<18} {u_non_str:<18} {time_str:<16}")
            else:
                u_nt_str = f"{res['weighted_u_nt']:.2f}"
                u_tr_str = f"{res['weighted_u_tr']:.2f}"
                u_non_str = f"{res['weighted_u_non']:.2f}"
                print(f"{i+1:<4} {desc:<35} {res['length']:<8} {res['n_structures']:<10} "
                      f"{res['n_components']:<6} {res['n_basins']:<12} "
                      f"{u_nt_str:<12} {u_tr_str:<12} {u_non_str:<12} {res['elapsed']:<10.1f}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {total_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()