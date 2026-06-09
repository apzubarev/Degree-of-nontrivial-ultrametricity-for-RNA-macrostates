
"""
Calculation of the degree of nontrivial ultrametricity for RNA macrostates.
PHYSICALLY RIGOROUS APPROACH: distance between basins via spectral
decomposition of the transition rate matrix (Mahalanobis distance
in the space of eigenvectors of the symmetrized matrix K).

METHOD:
1. A transition rate matrix K is constructed between all structures
   (N x N, where N ~ 2000) based on the Kramers formula.
2. K is symmetrized taking into account detailed balance.
3. The m smallest eigenvalues in magnitude and corresponding
   eigenvectors are computed (Lanczos method for sparse matrices).
4. Automatic filtering of noise modes is performed by searching for
   a spectral gap: if the ratio |lambda_k| / |lambda_{k-1}| exceeds
   a threshold (default 10^6), modes with indices < k are discarded
   as numerical noise.
5. Each basin of attraction is represented by a characteristic
   vector chi_A in the space of structures.
6. The distance between basins A and B is defined as the weighted
   Euclidean distance between projections of chi_A and chi_B onto
   eigenvectors (Mahalanobis distance).
7. The resulting distance matrix is a metric and is tested for
   ultrametricity.

HANDLING OF DISCONNECTED GRAPHS:
Before constructing the K_sym matrix, the connectivity of the structure
graph is checked. If the graph contains multiple connected components,
each component is processed separately: its own K_sym matrix is built,
spectral decomposition is performed, and ultrametricity is checked.
Components with fewer than 3 basins are skipped.
Components containing fewer than ALPHA_COMPONENT_THRESHOLD * N structures
are classified as noise and excluded from the calculation of f_inter and
the final spectral analysis.
IMPORTANT: This logic of processing ALL significant components with
subsequent weighted averaging is applied UNIFORMLY both in the main stage
and in all null hypothesis testing modes (nt_shuffle, energy_shuffle,
topo_shuffle). This guarantees statistical reliability of the results.

STATISTICAL MODE (NUM_STAT > 1):
When NUM_STAT > 1, NUM_STAT independent runs with different random
structure samples are performed for each sequence (seed varies:
RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
Runs are executed IN PARALLEL via multiprocessing.Pool to maximize
computational resource utilization.
Results are averaged, and the final table displays mean values and
standard deviations (mean +/- std). Integer quantities (number of
structures, basins, connected components) are rounded to integers.

OUTPUT MODES:
VERBOSE = True  -- full log (steps, components, spectral analysis).
VERBOSE = False -- brief log: sequence header and parameters are printed
                   once, then only RUN/COMPLETED, followed by statistics block.

NULL HYPOTHESIS TESTING (NULL_MODEL_TYPE):
Testing is performed by comparing the real system with null models
differing in the degree of "randomness". All null models are executed
IN PARALLEL and process ALL significant connected components:

'none'            : Program runs in normal mode without tests.

'full_analysis'   : (RECOMMENDED) FULL MECHANISM ANALYSIS.
                    Automatically performs TWO independent tests:
                    1. Energy Shuffle: Graph preservation + energy shuffling.
                       Shows contribution of pure graph topology.
                    2. Topo Shuffle: Configuration model (edge rewiring
                       preserving vertex degrees) + energy shuffling.
                       Shows baseline chaos level.
                    Outputs TWO separate summary tables for each test.

'topo_shuffle'    : CONFIGURATION MODEL.
                    Graph edge rewiring via double_edge_swap while
                    preserving vertex degree sequence + energy shuffling.
                    Destroys topological correlations while preserving
                    mobility distribution. Basins are re-identified for
                    ALL significant components.

'energy_shuffle'  : (WEAK RANDOMNESS / TOPOLOGICAL ORDER)
                    Neighborhood graph is fully preserved (including all
                    topological correlations), but vertex energies are
                    randomly shuffled. Basins are re-identified for these
                    random energies in ALL significant components. Allows
                    isolating the contribution of PURE GRAPH TOPOLOGY to
                    ultrametricity.

'nt_shuffle'      : NUCLEOTIDE SHUFFLING (BIOLOGICAL CONTROL).
                    Random permutations of the original RNA sequence are
                    generated preserving nucleotide composition. For each
                    permutation, all stages are fully re-executed: structure
                    generation, graph construction, search for ALL significant
                    components and basins, spectral analysis. This is the
                    strictest test checking whether ultrametricity is due to
                    specific nucleotide order. VERY TIME-CONSUMING.

'random_basins'   : (GEOMETRIC CONTROL)
                    Real spectrum of K_sym matrix is preserved, but structures
                    are randomly partitioned into basins of the same sizes.
                    Checks whether ultrametricity is an artifact of the
                    geometry of the high-dimensional eigenvector space.
                    Does not affect graph or energies.

RNA ENSEMBLE EXPECTATION (EXPECTATION_BY_RNA = True):
If NULL_MODEL_TYPE = 'none' and EXPECTATION_BY_RNA = True, a row
"MEAN OVER ALL RNAs" is added to the end of the summary table, containing
mean values and STD of all metrics across the entire set of sequences.

ADVANTAGES:
- Accounts for all possible transition paths (via spectral decomposition).
- Context-independent (distance between A and B is determined only by them,
  not by presence of other basins).
- Symmetric and guaranteed to be a metric.
- Automatically filters numerical noise via spectral gap detection.
- Correctly handles disconnected structure graphs.
- Computational complexity O(m*N*E + K^2*m), allowing processing of
  N ~ 2000 structures and K ~ 100 basins in seconds.
- Full parallelization of main stage and all null models.
- Unified methodology for component processing in all modes.

STRUCTURE GENERATION MODE:
Stochastic sampling (pbacktrack) from Gibbs distribution.

Dependencies: pip install viennarna numpy scipy biopython
"""

import RNA
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from itertools import combinations
from collections import defaultdict
from math import comb
import warnings
import time
import textwrap
import os
import glob
import gc
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL VARIABLES FOR PARALLEL NEIGHBOR GENERATION (OPTIMIZED)
# ============================================================================

_INDEX_MAP = None
"""Global variable: dictionary {bitmask: index} for O(1) lookup in workers."""

_CONFLICT_MASKS = None
"""Global variable: list of conflict bitmasks for each allowed pair."""

_BIT = None
"""Global variable: precomputed powers of two [1<<0, 1<<1, ...] for acceleration."""

_P = None
"""Global variable: total number of allowed pairs."""

# ============================================================================
# USER PARAMETERS
# ============================================================================

# --- Data source parameters ---

FASTA_RNA = True
"""
Mode for loading RNA sequences from FASTA files.
  True  -- scan current folder for *.fasta files,
           load all sequences, sort by length.
  False -- use sequence from RNA_SEQUENCE.
Recommended value: True (for research work).
"""

RNA_SEQUENCE = "ACATCAATCCACCACTCTTTCTCTTTAAAAAGAGTAGACCCAGGAACCGAAATTCTTTACCAAATTAAAAAA"
"""
Primary RNA structure. Used only when FASTA_RNA = False.
Allowed characters: A, U, G, C (uppercase, T is automatically replaced by U).
Recommended length: 50-200 nucleotides.
"""

# --- Temperature and energy parameters ---

TEMPERATURE_CELSIUS = 37.0
"""
Temperature in degrees Celsius.
Affects Boltzmann weights and transition probabilities.
  Low (< 20C): deep basins, rare transitions.
  High (> 60C): smoothed landscape, fast transitions.
Recommended value: 37.0 (physiological temperature).
Valid range: 0.0 - 100.0.
"""

ENERGY_WINDOW = 50.0
"""
Energy window (kcal/mol) relative to minimum free energy (MFE).
During stochastic sampling, structures with energy > MFE + ENERGY_WINDOW
are discarded. If set to "inf", no window is applied.
  Small window (1-5 kcal/mol): only most stable structures,
    may be insufficient for analysis.
  Large window (> 15 kcal/mol): many structures, sparse graph,
    computation time increases.
Recommended value: 10.0.
Valid range: positive number or "inf".
"""

# --- Structure generation parameters ---

MAX_STRUCTURES = 100000
"""
Maximum number of generated secondary structures (microstates).
Generation stops when the number of unique structures within the specified
energy window reaches this value.
  Few (100-500): fast but statistically poor analysis.
  Many (> 10000): complete landscape picture, but slow.
Recommended value: 2000-5000.
Valid range: 100 - 20000.
"""

MIN_HAIRPIN_LEN = 3
"""
Minimum number of unpaired nucleotides in a hairpin loop.
Defines condition: j - i - 1 >= MIN_HAIRPIN_LEN.
  Standard value: 3 (steric constraint).
  Value 0 disables constraint (unphysical).
Recommended value: 3.
Valid range: 0 - 10.
"""

RANDOM_SEED = 43
"""
Initial seed for random number generator.
Ensures reproducibility of results.
When NUM_STAT > 1, seed varies: RANDOM_SEED, RANDOM_SEED+1, ...
Recommended value: 42 (or any integer).
Valid range: any integer.
"""

# --- Basin of attraction parameters ---

MAX_MACROSTATES_ANALYSIS = 500
"""
Maximum number of basins of attraction participating in final analysis.
If more remain after filtering, basins with largest partition functions Z
are retained.
  Few (10-30): fast, but may lack triplet statistics.
  Many (> 200): more triplets for analysis, but slower (K^3 for spectrum).
Recommended value: 100.
Valid range: 3 - 500.
"""

MIN_MACROSTATE_SIZE = 5
"""
Minimum basin size (number of included structures).
Smaller basins are considered statistically insignificant.
  Value 1: all basins included, including isolated structures.
  Value 5-10: small artifactual basins filtered out.
Recommended value: 5.
Valid range: 1 - 100.
"""

# --- Connected component filtering parameter ---

ALPHA_COMPONENT_THRESHOLD = 0.001
"""
Relative threshold for classifying connected components of the structure
graph as significant or noise. A component is considered significant if it
contains at least max(3, ALPHA_COMPONENT_THRESHOLD * N) structures, where N
is the total number of unique structures in the sample. Noise components are
excluded from f_inter calculation and final spectral analysis.
  Small value (0.001): conservative, artifactual components may remain.
  Large value (0.05): aggressive, real small families may be lost.
Recommended value: 0.01.
Valid range: 0.001 - 0.1.
"""

# --- Spectral analysis parameters ---

NUM_EIGENMODES = 50
"""
Number of eigenmodes (eigenvalues and eigenvectors) requested for spectral
decomposition. After automatic noise mode filtering, actual number of used
modes may be smaller.
  Few (5-10): fast, but fine landscape structure information is lost.
  Many (> 100): more accurate, but slower (scales linearly).
  Constraint: must be strictly less than number of structures.
Recommended value: 50.
Valid range: 5 - 200 (but no more than N-2, where N is number of structures).
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for detecting spectral gap between noise and physical modes.
If ratio |lambda_k| / |lambda_{k-1}| > SPECTRAL_GAP_THRESHOLD, modes with
indices < k are considered numerical noise and discarded.
  Large threshold (10^8): conservative, weak physical modes may be lost.
  Small threshold (10^2): aggressive, noise modes may remain.
Recommended value: 1e6.
Valid range: 1e2 - 1e12.
"""

FREQUENCY_PREFACTOR = 1.0
"""
Frequency prefactor nu_0 in Kramers formula (in arbitrary units).
Affects absolute scale of matrix K, but does not affect eigenvectors or
relative distances between basins (changing nu_0 multiplies all lambda_k by
a constant, which cancels out in Mahalanobis distance).
Recommended value: 1.0 (leave unchanged).
Valid range: any positive number.
"""

EIGS_MAXITER = 50000
"""
Maximum number of iterations for Lanczos algorithm (ARPACK) when computing
eigenvalues of K_sym matrix. Increasing this parameter improves convergence
for matrices with dense spectrum near zero, but increases computation time.
Recommended value: 50000.
Valid range: 1000 - 200000.
"""

EIGS_SIGMA = 1e-10
"""
Shift sigma for Lanczos algorithm when searching for eigenvalues near zero.
Value should be positive and sufficiently small to avoid distorting the
spectrum of physical modes (which have |lambda| >= 10^-4), but sufficiently
large to avoid numerical singularity when solving (K_sym - sigma*I)x = b.
  Too small (10^-15): risk of numerical singularity.
  Too large (10^-3): distorts spectrum.
Recommended value: 1e-10.
Valid range: 1e-12 - 1e-6.
"""

# --- Ultrametricity check parameters ---

ULTRAMETRIC_EPSILON = 0.05
"""
Relative tolerance epsilon for approximate ultrametricity check.
Two largest sides of triangle are considered equal if
(d_max - d_mid) / d_mid <= epsilon.
  Must be strictly less than ULTRAMETRIC_DELTA.
  At epsilon = 0: exact equality required (almost unattainable).
  At epsilon > 0.1: many false positive classifications.
Recommended value: 0.05.
Valid range: 0.0 - 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference delta between smaller and middle sides of
triangle for classification as nontrivially ultrametric:
(d_mid - d_min) / d_mid > delta.
  Must be strictly greater than ULTRAMETRIC_EPSILON.
  At small delta: equilateral triangles erroneously classified as
    nontrivially ultrametric.
  At large delta: almost no nontrivially ultrametric triplets remain.
Recommended value: 0.1.
Valid range: 0.01 - 0.50.
"""

# --- Numerical precision parameters ---

EPS_COMPARISON = 1e-9
"""
Threshold for comparing floating-point numbers (energies, distances).
Used for checking strict inequalities in plateau conditions, local minima,
and triangle classification.
  Too small (< 1e-12): risk of false distinction due to rounding noise.
  Too large (> 1e-6): risk of merging distinct states.
Recommended value: 1e-9.
Valid range: 1e-12 - 1e-6.
"""

# --- Computational resource parameters ---

NUM_WORKERS = None
"""
Number of parallel processes for neighbor structure generation and
execution of NUM_STAT runs / null models.
  None: automatically use all available CPU cores.
  1: single-threaded mode (for debugging).
  N: use exactly N processes.
Recommended value: None.
Valid range: 1 - cpu_count().
"""

VERBOSE = False
"""
Verbose output mode.
  True: output all intermediate results (basin sizes, transition statistics,
    triangle distribution).
  False: only final results (brief log).
Recommended value: True (for research purposes).
"""

# --- Statistical analysis parameter ---

NUM_STAT = 1
"""
Number of statistical trials (independent runs) for each RNA sequence.
Runs are executed IN PARALLEL.
  NUM_STAT = 1: single run, result without deviation.
  NUM_STAT > 1: NUM_STAT runs performed with different seeds
    (RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
    Results are averaged, output as mean +/- STD.
    Integer quantities (number of structures, basins, components)
    are rounded to integers.
Recommended value: 1.
Valid range: 1 - 100.
"""

# --- Null hypothesis testing parameters ---

NULL_MODEL_TYPE = 'full_analysis'
"""
Type of null model for testing hypothesis about origin of ultrametricity.
All models are executed IN PARALLEL and process ALL significant components.
  'none'          : No null hypothesis testing.
  'full_analysis' : (RECOMMENDED) Full mechanism analysis.
                    Performs TWO tests: energy_shuffle and topo_shuffle.
                    Outputs two separate summary tables.
  'topo_shuffle'  : Configuration model. Graph edge rewiring
                    (double_edge_swap preserving vertex degrees) + energy
                    shuffling. Destroys topological correlations while
                    preserving mobility distribution.
  'energy_shuffle': Neighborhood graph preserved (including all topological
                    correlations), energies shuffled. Contribution of pure
                    graph topology to ultrametricity.
  'nt_shuffle'    : Nucleotide shuffling. Complete regeneration of
                    structures and graph for random sequence permutations.
                    Strictest biological control.
  'random_basins' : Geometric control. Real spectrum, random basins of
                    same sizes. Check for space artifacts.
Recommended value: 'full_analysis'.
"""

NUM_NULL_SAMPLES = 20
"""
Number of realizations for each null hypothesis test.
Executed IN PARALLEL via multiprocessing.Pool.
For 'random_basins', can set 100-500 (very fast).
For 'energy_shuffle', 'topo_shuffle', 20-30 recommended.
For 'nt_shuffle', 5-10 recommended (very slow, full recalculation).
In 'full_analysis' mode, this count applies to both tests.
"""

NUM_EDGE_SWAPS_MULTIPLIER = 10
"""
Multiplier for number of edge swaps during graph rewiring
(topo_shuffle and full_analysis modes).
Number of swaps = NUM_EDGE_SWAPS_MULTIPLIER * |E|.
  Small (1-3): fast mixing, but topological correlations may not fully break.
  Medium (5-10): good balance of speed and mixing quality.
  Large (>20): thorough correlation destruction, but slower.
Recommended value: 10.
Valid range: 1 - 100.
"""

# --- RNA ensemble averaging parameters ---

EXPECTATION_BY_RNA = False
"""
Mode for outputting summary statistics across all sequences.
Works only when NULL_MODEL_TYPE = 'none'.
  False: program runs unchanged.
  True:  a row with mean values and STD of all metrics across entire RNA set
         is added to end of summary table. When NUM_STAT > 1, averaging is
         performed over all runs of all sequences (grand mean and grand STD).
Recommended value: False (enable for generalized assessment).
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

R_KCAL = 0.001987204259  # Gas constant in kcal/(mol*K) (R = N_A * k_B)


# ============================================================================
# OPTIMIZATION: BITMASKS AND PRECOMPUTATION OF ALLOWED PAIRS + CONFLICTS
# ============================================================================

def precompute_allowed_pairs_and_conflicts(seq_len, sequence, min_hairpin_len, comp_map):
    """
    Precomputes list of all allowed pairs and conflict matrix between them.
    Conflicts are encoded as bitmasks for O(1) checking.

    Returns:
        allowed (list): list of pairs (i, j)
        pair_to_idx (dict): mapping of pair to its index
        conflict_masks (list): conflict bitmasks for each pair
        bit (list): precomputed powers of two
        P (int): number of allowed pairs
    """
    allowed = []
    for i in range(seq_len):
        for j in range(i + min_hairpin_len + 1, seq_len):
            if (sequence[i], sequence[j]) in comp_map:
                allowed.append((i, j))

    P = len(allowed)
    pair_to_idx = {pair: idx for idx, pair in enumerate(allowed)}

    # Precompute powers of two for faster bitwise operations
    bit = [1 << i for i in range(P)]

    # Precompute conflict masks
    conflict_masks = [0] * P
    for idx1 in range(P):
        i1, j1 = allowed[idx1]
        mask = 0
        for idx2 in range(P):
            if idx1 == idx2:
                continue
            i2, j2 = allowed[idx2]
            # Check for shared nucleotides or crossing (pseudoknot)
            if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
                mask |= bit[idx2]
            elif (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                mask |= bit[idx2]
        conflict_masks[idx1] = mask

    return allowed, pair_to_idx, conflict_masks, bit, P


# ============================================================================
# STRUCTURE HANDLING FUNCTIONS
# ============================================================================

def dotbracket_to_pairs(structure):
    """
    Converts dot-bracket structure to set of base pairs.
    Used for deduplication.
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


def dotbracket_to_bitmask(structure, pair_to_idx):
    """
    Converts dot-bracket structure to bitmask of allowed pair indices.
    Returns mask and tuple of set bits (for fast iteration).
    """
    mask = 0
    set_bits = []
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            u, v = (j, i) if j < i else (i, j)
            idx = pair_to_idx.get((u, v))
            if idx is not None:
                mask |= (1 << idx)
                set_bits.append(idx)
    return mask, tuple(set_bits)


def deduplicate_structures(structures, energies, verbose=True):
    """
    Removes duplicate structures (with identical pair sets).
    """
    unique_pairs = {}
    for s, e in zip(structures, energies):
        pairs = dotbracket_to_pairs(s)
        if pairs not in unique_pairs or e < unique_pairs[pairs][1]:
            unique_pairs[pairs] = (s, e)

    new_structures = [s for s, e in unique_pairs.values()]
    new_energies = [e for s, e in unique_pairs.values()]

    if verbose and len(new_structures) < len(structures):
        print(f"  Duplicates removed: {len(structures) - len(new_structures)}")

    return new_structures, np.array(new_energies)


# ============================================================================
# OPTIMIZATION: PARALLEL NEIGHBOR GENERATION (BITMASK IPC)
# ============================================================================

def _pool_initializer_bitmask(index_map, conflict_masks, bit, P):
    """
    Initializer for Pool processes.
    Sets module-level global variables for O(1) access.
    """
    global _INDEX_MAP, _CONFLICT_MASKS, _BIT, _P
    _INDEX_MAP = index_map
    _CONFLICT_MASKS = conflict_masks
    _BIT = bit
    _P = P


def _generate_neighbors_worker_bitmask(args):
    """
    Worker function for parallel neighbor generation.
    Accepts (idx, mask, set_bits), returns (idx, list_of_neighbor_indices).
    This radically reduces IPC overhead.
    """
    idx, mask, set_bits = args
    neighbors = []

    # Operation 1: Remove existing pair
    for idx_out in set_bits:
        new_mask = mask & ~_BIT[idx_out]
        nb_idx = _INDEX_MAP.get(new_mask)
        if nb_idx is not None:
            neighbors.append(nb_idx)

    # Operation 2: Add new pair
    for idx_in in range(_P):
        if not (mask & _BIT[idx_in]):
            # O(1) conflict check via bitwise AND
            if (mask & _CONFLICT_MASKS[idx_in]) == 0:
                new_mask = mask | _BIT[idx_in]
                nb_idx = _INDEX_MAP.get(new_mask)
                if nb_idx is not None:
                    neighbors.append(nb_idx)

    # Operation 3: Shift pair (remove + add)
    for idx_out in set_bits:
        temp_mask = mask & ~_BIT[idx_out]
        for idx_in in range(_P):
            if idx_in == idx_out:
                continue
            if not (temp_mask & _BIT[idx_in]):
                if (temp_mask & _CONFLICT_MASKS[idx_in]) == 0:
                    new_mask = temp_mask | _BIT[idx_in]
                    nb_idx = _INDEX_MAP.get(new_mask)
                    if nb_idx is not None:
                        neighbors.append(nb_idx)

    return (idx, neighbors)


def _build_neighbor_graph_local(struct_masks, struct_set_bits, index_map, conflict_masks, bit, P):
    """
    Local version of graph construction, not using global variables.
    Necessary for correct nt_shuffle operation in multiprocessing,
    where each worker has its own sequence and graph.
    """
    n_structures = len(struct_masks)
    neighbors_list = [set() for _ in range(n_structures)]

    for idx in range(n_structures):
        mask = struct_masks[idx]
        set_bits = struct_set_bits[idx]

        # Operation 1: Remove existing pair
        for idx_out in set_bits:
            new_mask = mask & ~bit[idx_out]
            nb_idx = index_map.get(new_mask)
            if nb_idx is not None:
                neighbors_list[idx].add(nb_idx)

        # Operation 2: Add new pair
        for idx_in in range(P):
            if not (mask & bit[idx_in]):
                if (mask & conflict_masks[idx_in]) == 0:
                    new_mask = mask | bit[idx_in]
                    nb_idx = index_map.get(new_mask)
                    if nb_idx is not None:
                        neighbors_list[idx].add(nb_idx)

        # Operation 3: Shift pair (remove + add)
        for idx_out in set_bits:
            temp_mask = mask & ~bit[idx_out]
            for idx_in in range(P):
                if idx_in == idx_out:
                    continue
                if not (temp_mask & bit[idx_in]):
                    if (temp_mask & conflict_masks[idx_in]) == 0:
                        new_mask = temp_mask | bit[idx_in]
                        nb_idx = index_map.get(new_mask)
                        if nb_idx is not None:
                            neighbors_list[idx].add(nb_idx)

    return neighbors_list


# ============================================================================
# GRAPH CONSTRUCTION AND ANALYSIS FUNCTIONS
# ============================================================================

def generate_structures_stochastic(seq, temp_celsius, max_structures, energy_window, verbose=True):
    """
    Generation of secondary structure set by stochastic sampling
    from Boltzmann ensemble (pbacktrack) with energy cutoff.
    """
    RNA.cvar.temperature = temp_celsius
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)

    (mfe_struct, mfe) = fc.mfe()
    fc.pf()

    if isinstance(energy_window, str) and energy_window.lower() == "inf":
        energy_cutoff = float('inf')
    else:
        energy_cutoff = mfe + float(energy_window)

    structures = []
    energies_list = []
    seen = set()

    if mfe <= energy_cutoff + EPS_COMPARISON:
        structures.append(mfe_struct)
        energies_list.append(mfe)
        seen.add(mfe_struct)

    batch_size = min(max_structures, 500)
    max_batches = (max_structures * 10) // batch_size + 1

    for batch in range(max_batches):
        if len(structures) >= max_structures:
            break
        try:
            for struct in fc.pbacktrack(batch_size):
                if len(structures) >= max_structures or not struct or struct in seen:
                    continue
                energy = fc.eval_structure(struct)
                if energy <= energy_cutoff + EPS_COMPARISON:
                    seen.add(struct)
                    structures.append(struct)
                    energies_list.append(energy)
        except Exception:
            break

    if verbose:
        print(f"  Stochastic sampling: generated {len(structures)} unique structures "
              f"(requested {max_structures})")

    return structures, np.array(energies_list)


def build_neighbor_graph_bitmask(struct_masks, struct_set_bits, index_map, conflict_masks, bit, P, num_workers=None, verbose=True):
    """
    Builds neighborhood graph based on neighbor generation via bitmasks.
    Uses optimized IPC (returns only indices).
    """
    n_workers = num_workers if num_workers else cpu_count()
    n_structures = len(struct_masks)
    neighbors_list = [set() for _ in range(n_structures)]

    if n_workers > 1:
        if verbose:
            print(f"  Using {n_workers} processes for neighbor generation (Bitmask O(1) IPC)")
        indexed_args = [(idx, struct_masks[idx], struct_set_bits[idx]) for idx in range(n_structures)]
        with Pool(n_workers, initializer=_pool_initializer_bitmask,
                  initargs=(index_map, conflict_masks, bit, P)) as pool:
            # Increased chunksize to reduce dispatch overhead
            for idx, neighbor_indices in pool.imap_unordered(
                _generate_neighbors_worker_bitmask, indexed_args, chunksize=50
            ):
                neighbors_list[idx].update(neighbor_indices)
    else:
        _pool_initializer_bitmask(index_map, conflict_masks, bit, P)
        for idx in range(n_structures):
            _, neighbor_indices = _generate_neighbors_worker_bitmask((idx, struct_masks[idx], struct_set_bits[idx]))
            neighbors_list[idx].update(neighbor_indices)

    return neighbors_list


def find_connected_components(neighbors_list):
    """
    Finds connected components of the structure graph.
    """
    n = len(neighbors_list)
    visited = [False] * n
    components = []

    for start in range(n):
        if not visited[start]:
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

    components.sort(key=len, reverse=True)
    return components


def compute_gradient_basins(energies, neighbors_list, verbose=True):
    """
    Identifies basins of attraction (gradient basins) with correct
    plateau handling.
    """
    n = len(energies)
    candidate_set = {i for i in range(n) if not any(energies[nb] < energies[i] - EPS_COMPARISON for nb in neighbors_list[i])}

    visited_candidate, attraction_points = set(), []
    for v in candidate_set:
        if v not in visited_candidate:
            component, stack = [], [v]
            visited_candidate.add(v)
            while stack:
                u = stack.pop()
                component.append(u)
                for nb in neighbors_list[u]:
                    if nb in candidate_set and nb not in visited_candidate and abs(energies[u] - energies[nb]) < EPS_COMPARISON:
                        visited_candidate.add(nb)
                        stack.append(nb)
            attraction_points.append(component)

    attraction_id = {v: idx for idx, comp in enumerate(attraction_points) for v in comp}
    basin_of = [-1] * n

    def find_basin(i):
        if basin_of[i] != -1: return basin_of[i]
        if i in attraction_id:
            basin_of[i] = attraction_id[i]
            return attraction_id[i]
        neighbors = list(neighbors_list[i])
        if not neighbors or min(neighbors, key=lambda x: (energies[x], x)) == i or energies[min(neighbors, key=lambda x: (energies[x], x))] >= energies[i] - EPS_COMPARISON:
            new_id = len(attraction_points)
            attraction_points.append([i])
            attraction_id[i] = new_id
            basin_of[i] = new_id
            return new_id
        basin = find_basin(min(neighbors, key=lambda x: (energies[x], x)))
        basin_of[i] = basin
        return basin

    for i in range(n): find_basin(i)
    basins_dict = defaultdict(list)
    for idx, b in enumerate(basin_of): basins_dict[b].append(idx)

    basins = [(attraction_points[b][0], indices) for b, indices in basins_dict.items()]
    basins.sort(key=lambda x: energies[x[0]])

    if verbose:
        print(f"  Number of macrostates (basins): {len(basins)}")

    return basins


def build_transition_rate_matrix(energies, neighbors_list, temp_kelvin, nu0):
    """
    Constructs symmetrized transition rate matrix K_sym.
    """
    N = len(energies)
    RT = R_KCAL * temp_kelvin

    K_sym = lil_matrix((N, N), dtype=np.float64)
    row_sums = np.zeros(N, dtype=np.float64)

    for p in range(N):
        G_p = energies[p]
        for q in neighbors_list[p]:
            if q > p:
                rate = nu0 * np.exp(-abs(G_p - energies[q]) / (2.0 * RT))
                K_sym[p, q] = rate
                K_sym[q, p] = rate
                row_sums[p] += rate
                row_sums[q] += rate

    for p in range(N):
        K_sym[p, p] = -row_sums[p]

    return K_sym.tocsr()


def filter_eigenvalues_by_gap(eigenvalues, eigenvectors, gap_threshold):
    """
    Automatically finds spectral gap and filters out noise modes.
    """
    idx_sorted = np.argsort(np.abs(eigenvalues))
    sorted_vals = eigenvalues[idx_sorted]
    sorted_vecs = eigenvectors[:, idx_sorted]

    abs_vals = np.abs(sorted_vals)
    num_noise = 1

    for k in range(1, len(abs_vals)):
        if abs_vals[k-1] < 1e-30:
            ratio = float('inf') if abs_vals[k] > 1e-30 else 1.0
        else:
            ratio = abs_vals[k] / abs_vals[k-1]

        if ratio > gap_threshold:
            num_noise = k
            break
    else:
        num_noise = 1

    return sorted_vals[num_noise:], sorted_vecs[:, num_noise:], num_noise


def filter_macrostates_spectral(basins, Z, min_size, max_macrostates, verbose=True):
    """
    Filters macrostates by size and statistical significance.
    """
    valid = [i for i, (_, indices) in enumerate(basins) if len(indices) >= min_size]

    if verbose:
        print(f"  Excluded macrostates with size < {min_size}: {len(basins) - len(valid)}")

    if len(valid) > max_macrostates:
        valid.sort(key=lambda i: Z[i], reverse=True)
        valid = valid[:max_macrostates]
        if verbose:
            print(f"  Retained macrostates with largest Z: {len(valid)} (out of {len(basins)})")
    else:
        valid.sort(key=lambda i: Z[i], reverse=True)
        if verbose:
            print(f"  Retained macrostates: {len(valid)}")

    return [basins[i] for i in valid], {old: new for new, old in enumerate(valid)}


def compute_spectral_distance(K_sym, basins, num_modes_requested, temp_kelvin, gap_threshold, eigs_maxiter, eigs_sigma, verbose=True):
    """
    Computes Mahalanobis distance between basins.
    Also returns number of found physical modes for logging.
    """
    N = K_sym.shape[0]
    K_basins = len(basins)
    num_modes_requested = min(num_modes_requested, N - 1)
    ncv = min(2 * num_modes_requested + 10, N)

    eigenvalues, eigenvectors = None, None
    last_error = None

    try:
        eigenvalues, eigenvectors = eigsh(K_sym, k=num_modes_requested, which='SM', return_eigenvectors=True, maxiter=eigs_maxiter, ncv=ncv, tol=1e-8)
    except Exception as e:
        last_error = e
        try:
            eigenvalues, eigenvectors = eigsh(K_sym, k=num_modes_requested, which='LM', sigma=eigs_sigma, return_eigenvectors=True, maxiter=eigs_maxiter, ncv=ncv)
        except Exception as e2:
            last_error = e2

    if eigenvalues is None:
        raise RuntimeError(f"Failed to compute eigenvalues. Last error: {last_error}")

    eigenvalues_filtered, eigenvectors_filtered, num_noise = filter_eigenvalues_by_gap(eigenvalues, eigenvectors, gap_threshold)
    num_phys = len(eigenvalues_filtered)

    if num_phys == 0:
        raise RuntimeError("All eigenmodes filtered as noise.")

    chi = np.zeros((K_basins, N), dtype=np.float64)
    for a, (_, indices) in enumerate(basins):
        chi[a, indices] = 1.0 / np.sqrt(len(indices))

    proj = chi @ eigenvectors_filtered
    weights = 1.0 / np.abs(eigenvalues_filtered)

    dist_matrix = np.zeros((K_basins, K_basins), dtype=np.float64)
    for a in range(K_basins):
        for b in range(a + 1, K_basins):
            diff = proj[a, :] - proj[b, :]
            dist_matrix[a, b] = np.sqrt(np.sum(weights * (diff ** 2)))
            dist_matrix[b, a] = dist_matrix[a, b]

    dist_matrix *= R_KCAL * temp_kelvin
    return dist_matrix, eigenvalues_filtered, eigenvectors_filtered, num_noise, num_phys


def classify_triangle(d1, d2, d3, eps, delta):
    """
    Triangle classification by ultrametricity.
    """
    if d1 == float('inf') or d2 == float('inf') or d3 == float('inf'):
        return 'non_ultrametric'

    d_min, d_mid, d_max = sorted([d1, d2, d3])

    if d_max <= EPS_COMPARISON:
        return 'trivial'

    if d_mid > EPS_COMPARISON:
        if (d_max - d_mid) / d_mid <= eps and (d_mid - d_min) / d_mid > delta:
            return 'nontrivial'

    if d_min > EPS_COMPARISON:
        if (d_max - d_min) / d_min <= eps:
            return 'trivial'
    elif d_max <= EPS_COMPARISON:
        return 'trivial'

    return 'non_ultrametric'


def compute_ultrametricity_score(dist_matrix, eps, delta):
    """Computation of ultrametricity degrees."""
    n = dist_matrix.shape[0]
    if n < 3:
        return 0.0, 0.0, 0.0, defaultdict(int)

    triplets = list(combinations(range(n), 3))
    if not triplets:
        return 0.0, 0.0, 0.0, defaultdict(int)

    counts = defaultdict(int)
    for i, j, k in triplets:
        counts[classify_triangle(dist_matrix[i, j], dist_matrix[i, k], dist_matrix[j, k], eps, delta)] += 1

    total = len(triplets)
    return (counts.get('nontrivial', 0) / total * 100,
            counts.get('trivial', 0) / total * 100,
            counts.get('non_ultrametric', 0) / total * 100, counts)


def _analyze_all_components(energies, neighbors_list, graph_components, n_total_structures,
                            temp_kelvin, min_basin_size, max_macrostates, num_modes,
                            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta,
                            alpha_threshold, verbose=False):
    """
    Universal function for analyzing ALL significant connected components.
    Applied UNIFORMLY in main stage and all null models.

    For each significant component:
    1. Structures and neighbors within component are extracted.
    2. Basins of attraction are identified.
    3. Filtered by size and partition function.
    4. K_sym is built and spectral decomposition performed.
    5. Ultrametricity is computed.

    Returns weighted averages u_nt, u_tr, u_non (weight = number of triplets),
    and auxiliary statistics including f_inter.
    """
    min_component_size = max(3, int(alpha_threshold * n_total_structures))

    global_basin_stats = []
    n_processed_components = 0
    n_significant_components = 0

    # For f_inter calculation: count triplets within each component
    intra_triplets_total = 0

    for comp_idx, comp_indices in enumerate(graph_components):
        if len(comp_indices) < min_component_size:
            continue
        n_significant_components += 1

        comp_set = set(comp_indices)
        comp_energies = energies[list(comp_indices)]
        old_to_local = {old: local for local, old in enumerate(comp_indices)}

        # Build neighbor list within component
        comp_neighbors = []
        for old_idx in comp_indices:
            local_nbs = {old_to_local[nb] for nb in neighbors_list[old_idx] if nb in comp_set}
            comp_neighbors.append(local_nbs)

        # Find basins
        comp_basins_raw = compute_gradient_basins(comp_energies, comp_neighbors, verbose=False)

        # Filter basins
        RT = R_KCAL * temp_kelvin
        Z = {i: sum(np.exp(-comp_energies[idx] / RT) for idx in indices)
             for i, (_, indices) in enumerate(comp_basins_raw)}

        filtered_basins, _ = filter_macrostates_spectral(
            comp_basins_raw, Z, min_basin_size, max_macrostates, verbose=False
        )

        if len(filtered_basins) < 3:
            continue

        n_processed_components += 1

        # Build K_sym and spectral analysis
        K_sym_comp = build_transition_rate_matrix(comp_energies, comp_neighbors, temp_kelvin, FREQUENCY_PREFACTOR)
        num_requested = min(num_modes, len(comp_indices) - 1)

        try:
            dist_matrix, _, _, num_noise, num_phys = compute_spectral_distance(
                K_sym_comp, filtered_basins, num_requested, temp_kelvin,
                gap_threshold, eigs_maxiter, eigs_sigma, verbose=False
            )
        except Exception:
            continue

        u_nt_comp, u_tr_comp, u_non_comp, counts_comp = compute_ultrametricity_score(
            dist_matrix, eps, delta
        )

        n_triplets = sum(counts_comp.values())
        n_basins_comp = len(filtered_basins)
        intra_triplets_total += n_triplets

        global_basin_stats.append({
            'u_nt': u_nt_comp, 'u_tr': u_tr_comp, 'u_non': u_non_comp,
            'num_triplets': n_triplets, 'num_basins': n_basins_comp,
            'num_phys_modes': num_phys, 'num_noise_modes': num_noise,
            'comp_idx': comp_idx
        })

    # Weighted averaging over all processed components
    total_triplets = sum(s['num_triplets'] for s in global_basin_stats)
    total_basins = sum(s['num_basins'] for s in global_basin_stats)

    if total_triplets > 0:
        weighted_u_nt = sum(s['u_nt'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
        weighted_u_tr = sum(s['u_tr'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
        weighted_u_non = sum(s['u_non'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
    else:
        weighted_u_nt = 0.0
        weighted_u_tr = 0.0
        weighted_u_non = 0.0

    # Calculate f_inter
    # f_inter = fraction of basin triplets belonging to DIFFERENT components
    # = 1 - (sum of intra-component triplets) / (total triplets)
    # Total triplets = C(total_basins, 3)
    if total_basins >= 3:
        total_possible_triplets = comb(total_basins, 3)
        if total_possible_triplets > 0:
            f_inter = 1.0 - (intra_triplets_total / total_possible_triplets)
        else:
            f_inter = 0.0
    else:
        f_inter = 0.0

    return {
        'weighted_u_nt': weighted_u_nt,
        'weighted_u_tr': weighted_u_tr,
        'weighted_u_non': weighted_u_non,
        'f_inter': f_inter,
        'n_significant_components': n_significant_components,
        'n_processed_components': n_processed_components,
        'total_basins': total_basins,
        'total_triplets': total_triplets,
        'global_basin_stats': global_basin_stats
    }


# ============================================================================
# WORKER FUNCTIONS FOR NULL HYPOTHESIS TESTING
# ============================================================================

def _double_edge_swap(edges_list, n_swaps, rng):
    """
    Performs double edge swap on edge list, strictly preserving vertex degrees.
    Destroys topological correlations of graph (clustering, modularity,
    assortativity), but preserves vertex degree sequence (conformational
    mobility distribution). This is basis of configuration model for null tests.
    """
    edges = list(edges_list)
    n_edges = len(edges)
    if n_edges < 2:
        return edges

    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    swaps_done = 0
    attempts = 0
    max_attempts = n_swaps * 100

    while swaps_done < n_swaps and attempts < max_attempts:
        attempts += 1
        i = rng.randint(0, n_edges)
        j = rng.randint(0, n_edges)
        if i == j:
            continue

        u, v = edges[i]
        x, y = edges[j]
        if u > v: u, v = v, u
        if x > y: x, y = y, x

        if len({u, v, x, y}) < 4:
            continue

        if rng.random() < 0.5:
            new_e1, new_e2 = (min(u, x), max(u, x)), (min(v, y), max(v, y))
        else:
            new_e1, new_e2 = (min(u, y), max(u, y)), (min(v, x), max(v, x))

        if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
            continue
        if new_e1 in edge_set or new_e2 in edge_set:
            continue

        edge_set.discard((u, v))
        edge_set.discard((x, y))
        edge_set.add(new_e1)
        edge_set.add(new_e2)
        edges[i], edges[j] = new_e1, new_e2
        swaps_done += 1

    return edges


def _random_basins_worker(args):
    """
    Control test: real spectrum, random basins of same sizes.
    Checks whether ultrametricity is artifact of high-dimensional
    eigenvector space geometry.
    """
    (worker_id, N_comp, basin_sizes, eigenvalues_filtered, eigenvectors_filtered, temp_kelvin, eps, delta, seed) = args
    rng = np.random.RandomState(seed)

    indices = np.arange(N_comp)
    rng.shuffle(indices)

    random_basins = []
    start = 0
    for size in basin_sizes:
        random_basins.append((start, indices[start:start+size].tolist()))
        start += size

    K_basins = len(random_basins)
    chi = np.zeros((K_basins, N_comp), dtype=np.float64)
    for a, (_, indices_b) in enumerate(random_basins):
        chi[a, indices_b] = 1.0 / np.sqrt(len(indices_b))

    proj = chi @ eigenvectors_filtered
    weights = 1.0 / np.abs(eigenvalues_filtered)

    dist_matrix = np.zeros((K_basins, K_basins), dtype=np.float64)
    for a in range(K_basins):
        for b in range(a + 1, K_basins):
            diff = proj[a, :] - proj[b, :]
            dist_matrix[a, b] = np.sqrt(np.sum(weights * (diff ** 2)))
            dist_matrix[b, a] = dist_matrix[a, b]
    dist_matrix *= R_KCAL * temp_kelvin

    u_nt, _, _, _ = compute_ultrametricity_score(dist_matrix, eps, delta)
    return u_nt


def _energy_shuffle_worker(args):
    """
    Null hypothesis test (energy_shuffle):
    1. Neighborhood graph fully preserved (all topological correlations).
    2. Vertex energies randomly shuffled.
    3. ALL significant connected components processed with weighted averaging.

    Shows contribution of PURE GRAPH TOPOLOGY to ultrametricity.
    Returns tuple (u_nt, u_tr, u_non).
    """
    (worker_id, real_energies, neighbors_list, graph_components, n_total_structures,
     temp_kelvin, min_basin_size, max_macrostates, num_modes,
     gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold, seed) = args

    rng = np.random.RandomState(seed)

    try:
        # Shuffle energies (graph NOT changed)
        shuffled_energies = rng.permutation(real_energies)

        # Analyze ALL significant components
        result = _analyze_all_components(
            shuffled_energies, neighbors_list, graph_components, n_total_structures,
            temp_kelvin, min_basin_size, max_macrostates, num_modes,
            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold
        )

        if result['total_triplets'] == 0:
            return None

        return (result['weighted_u_nt'], result['weighted_u_tr'], result['weighted_u_non'])

    except Exception:
        return None


def _config_model_worker(args):
    """
    Null hypothesis test (topo_shuffle / configuration model):
    1. Graph edge rewiring via double_edge_swap (preserving vertex degrees).
    2. Energy shuffling.
    3. ALL significant connected components processed with weighted averaging.

    MAXIMUM CHAOS model for given vertex degree distribution.
    Returns tuple (u_nt, u_tr, u_non).
    """
    (worker_id, real_energies, edges_list, degree_sequence, graph_components_unused,
     n_total_structures, temp_kelvin, min_basin_size, max_macrostates, num_modes,
     gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold,
     n_swaps, seed) = args

    rng = np.random.RandomState(seed)

    try:
        # 1. Rewire graph edges
        shuffled_edges = _double_edge_swap(edges_list, n_swaps, rng)

        # 2. Shuffle energies
        shuffled_energies = rng.permutation(real_energies)

        # 3. Build new neighbor list
        N = len(degree_sequence)
        new_neighbors_list = [set() for _ in range(N)]
        for u, v in shuffled_edges:
            new_neighbors_list[u].add(v)
            new_neighbors_list[v].add(u)

        # 4. Find connected components of new graph
        new_components = find_connected_components(new_neighbors_list)

        # 5. Analyze ALL significant components
        result = _analyze_all_components(
            shuffled_energies, new_neighbors_list, new_components, n_total_structures,
            temp_kelvin, min_basin_size, max_macrostates, num_modes,
            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold
        )

        if result['total_triplets'] == 0:
            return None

        return (result['weighted_u_nt'], result['weighted_u_tr'], result['weighted_u_non'])

    except Exception:
        return None


def _nt_shuffle_worker(args):
    """
    Null hypothesis test (nt_shuffle):
    1. Nucleotide shuffling preserving composition.
    2. Complete regeneration of structures, graph, components, basins, spectrum.
    3. ALL significant connected components processed with weighted averaging.

    Strictest biological control.
    Returns tuple (u_nt, u_tr, u_non).
    """
    (worker_id, original_seq, temp_kelvin, max_structures, energy_window,
     min_hairpin_len, min_basin_size, max_macrostates, num_modes,
     gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold, seed) = args

    rng = np.random.RandomState(seed)

    try:
        # 1. Shuffle nucleotides
        seq_list = list(original_seq)
        rng.shuffle(seq_list)
        shuffled_seq = ''.join(seq_list)

        # 2. Generate structures
        structures, energies = generate_structures_stochastic(
            shuffled_seq, TEMPERATURE_CELSIUS, max_structures, energy_window, verbose=False
        )

        if len(structures) < 2:
            return None

        structures, energies = deduplicate_structures(structures, energies, verbose=False)

        # 3. Precompute pairs and conflicts FOR SHUFFLED sequence
        comp_map = {('A', 'U'): True, ('U', 'A'): True, ('G', 'C'): True,
                    ('C', 'G'): True, ('G', 'U'): True, ('U', 'G'): True}
        allowed_pairs, pair_to_idx, conflict_masks, bit, P = precompute_allowed_pairs_and_conflicts(
            len(shuffled_seq), shuffled_seq, min_hairpin_len, comp_map
        )

        # 4. Convert to bitmasks
        struct_masks = []
        struct_set_bits = []
        index_map = {}
        for idx, s in enumerate(structures):
            mask, set_bits = dotbracket_to_bitmask(s, pair_to_idx)
            struct_masks.append(mask)
            struct_set_bits.append(set_bits)
            index_map[mask] = idx

        # 5. Build graph (local function, no global variables)
        neighbors = _build_neighbor_graph_local(
            struct_masks, struct_set_bits, index_map, conflict_masks, bit, P
        )

        del struct_masks, struct_set_bits, index_map
        gc.collect()

        # 6. Find ALL connected components
        graph_components = find_connected_components(neighbors)
        n_total = len(energies)

        # 7. Analyze ALL significant components
        result = _analyze_all_components(
            energies, neighbors, graph_components, n_total,
            temp_kelvin, min_basin_size, max_macrostates, num_modes,
            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold
        )

        if result['total_triplets'] == 0:
            return None

        return (result['weighted_u_nt'], result['weighted_u_tr'], result['weighted_u_non'])

    except Exception:
        return None


def run_null_hypothesis_tests(real_u_nt, comp_res, temp_kelvin, n_workers, original_seq=None, verbose=False):
    """
    Runs selected null hypothesis test depending on NULL_MODEL_TYPE.
    All tests are executed IN PARALLEL and process ALL significant components.
    Supports modes: random_basins, energy_shuffle, topo_shuffle, nt_shuffle, full_analysis.
    Mode full_analysis performs both tests (energy_shuffle and topo_shuffle)
    and returns results of both as dictionary with keys 'energy' and 'topo'.
    Also returns execution time for each test.
    """
    if NULL_MODEL_TYPE == 'none':
        return None, None

    n_cpus = n_workers if n_workers else cpu_count()

    null_stats, control_stats = None, None

    # Determine which tests to run
    run_energy = NULL_MODEL_TYPE in ('energy_shuffle', 'full_analysis')
    run_topo = NULL_MODEL_TYPE in ('topo_shuffle', 'full_analysis')
    run_nt = NULL_MODEL_TYPE == 'nt_shuffle'
    run_random = NULL_MODEL_TYPE == 'random_basins'

    # === RANDOM BASELINES TEST ===
    if run_random:
        print(f"\n--- CONTROL TEST: RANDOM BASINS (REAL SPECTRUM) ---")
        print(f"  Realizations: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")

        # For random_basins use data from comp_res (one component)
        basins = comp_res['basins']
        basin_sizes = [len(indices) for _, indices in basins]
        evals = comp_res['eigenvalues_filtered']
        evecs = comp_res['eigenvectors_filtered']
        N_comp = comp_res['comp_size']

        start_time = time.time()
        worker_args = [(i, N_comp, basin_sizes, evals, evecs, temp_kelvin,
                        ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, RANDOM_SEED + 20000 + i)
                       for i in range(NUM_NULL_SAMPLES)]

        with Pool(processes=n_cpus) as pool:
            results = pool.map(_random_basins_worker, worker_args)

        elapsed_test = time.time() - start_time

        u_nt_arr = np.array([r for r in results if r is not None])
        if len(u_nt_arr) > 0:
            control_stats = {
                'mean_u_nt': np.mean(u_nt_arr),
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'p_value': float(np.mean(u_nt_arr >= real_u_nt)),
                'elapsed': elapsed_test
            }
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:       {real_u_nt:.2f}%")
            print(f"  Random u_nt:     {control_stats['mean_u_nt']:.2f} +/- {control_stats['std_u_nt']:.2f}%")
            print(f"  p-value:         {control_stats['p_value']:.4f}")

    # === ENERGY SHUFFLE TEST ===
    energy_stats = None
    if run_energy:
        print(f"\n--- TEST 1: ENERGY SHUFFLE (GRAPH PRESERVED) ---")
        print(f"  Realizations: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Energy shuffling, processing ALL significant components)")

        # Data from comp_res: graph and components from first run
        real_energies = comp_res['all_energies']
        neighbors_list = comp_res['all_neighbors']
        graph_components = comp_res['all_components']
        n_total = comp_res['n_total_structures']
        num_modes = min(NUM_EIGENMODES, n_total - 1)

        start_time = time.time()
        worker_args = [
            (i, real_energies, neighbors_list, graph_components, n_total,
             temp_kelvin, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, num_modes,
             SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
             ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
             RANDOM_SEED + 30000 + i)
            for i in range(NUM_NULL_SAMPLES)
        ]

        with Pool(processes=n_cpus) as pool:
            results = pool.map(_energy_shuffle_worker, worker_args)

        elapsed_test = time.time() - start_time

        valid_results = [r for r in results if r is not None]

        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])

            energy_stats = {
                'mean_u_nt': np.mean(u_nt_arr),
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': float(np.mean(u_nt_arr >= real_u_nt)),
                'elapsed': elapsed_test
            }
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:       {real_u_nt:.2f}%")
            print(f"  EnergyShuffle u_nt:  {energy_stats['mean_u_nt']:.2f} +/- {energy_stats['std_u_nt']:.2f}%")
            print(f"  EnergyShuffle u_tr:  {energy_stats['mean_u_tr']:.2f} +/- {energy_stats['std_u_tr']:.2f}%")
            print(f"  EnergyShuffle u_non: {energy_stats['mean_u_non']:.2f} +/- {energy_stats['std_u_non']:.2f}%")
            print(f"  p-value:             {energy_stats['p_value']:.4f}")

    # === TOPO SHUFFLE TEST (configuration model) ===
    topo_stats = None
    if run_topo:
        print(f"\n--- TEST 2: CONFIGURATION MODEL + ENERGY SHUFFLE ---")
        print(f"  Realizations: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Edge rewiring + energy shuffling, processing ALL significant components)")

        real_energies = comp_res['all_energies']
        neighbors_list = comp_res['all_neighbors']
        n_total = comp_res['n_total_structures']
        num_modes = min(NUM_EIGENMODES, n_total - 1)

        # Extract edges from full graph
        edges_set = set()
        for u, nbs in enumerate(neighbors_list):
            for v in nbs:
                if u < v:
                    edges_set.add((u, v))
        edges_list = list(edges_set)

        # Compute degree sequence
        degree_sequence = np.zeros(n_total, dtype=int)
        for u, v in edges_list:
            degree_sequence[u] += 1
            degree_sequence[v] += 1

        n_swaps = NUM_EDGE_SWAPS_MULTIPLIER * len(edges_list)
        print(f"  Edges: {len(edges_list)}, Swaps per realization: {n_swaps}")

        start_time = time.time()
        worker_args = [
            (i, real_energies, edges_list, degree_sequence, None, n_total,
             temp_kelvin, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, num_modes,
             SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
             ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
             n_swaps, RANDOM_SEED + 40000 + i)
            for i in range(NUM_NULL_SAMPLES)
        ]

        with Pool(processes=n_cpus) as pool:
            results = pool.map(_config_model_worker, worker_args)

        elapsed_test = time.time() - start_time

        valid_results = [r for r in results if r is not None]

        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])

            topo_stats = {
                'mean_u_nt': np.mean(u_nt_arr),
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': float(np.mean(u_nt_arr >= real_u_nt)),
                'elapsed': elapsed_test
            }
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:       {real_u_nt:.2f}%")
            print(f"  ConfigModel u_nt:    {topo_stats['mean_u_nt']:.2f} +/- {topo_stats['std_u_nt']:.2f}%")
            print(f"  ConfigModel u_tr:    {topo_stats['mean_u_tr']:.2f} +/- {topo_stats['std_u_tr']:.2f}%")
            print(f"  ConfigModel u_non:   {topo_stats['mean_u_non']:.2f} +/- {topo_stats['std_u_non']:.2f}%")
            print(f"  p-value:             {topo_stats['p_value']:.4f}")

    # === NT SHUFFLE TEST (nucleotide shuffling) ===
    nt_stats = None
    if run_nt:
        if original_seq is None:
            print("\n  ERROR: Original sequence required for nt_shuffle mode.")
            return None, None

        print(f"\n--- TEST: NUCLEOTIDE SHUFFLE (FULL RECALCULATION) ---")
        print(f"  Realizations: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Complete regeneration + processing ALL significant components)")
        print(f"  WARNING: This test may take a long time!")

        start_time = time.time()
        worker_args = [
            (i, original_seq, temp_kelvin, MAX_STRUCTURES, ENERGY_WINDOW,
             MIN_HAIRPIN_LEN, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS,
             NUM_EIGENMODES, SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
             ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
             RANDOM_SEED + 50000 + i)
            for i in range(NUM_NULL_SAMPLES)
        ]

        with Pool(processes=n_cpus) as pool:
            results = pool.map(_nt_shuffle_worker, worker_args)

        elapsed_test = time.time() - start_time

        valid_results = [r for r in results if r is not None]

        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])

            nt_stats = {
                'mean_u_nt': np.mean(u_nt_arr),
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': float(np.mean(u_nt_arr >= real_u_nt)),
                'elapsed': elapsed_test
            }
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:       {real_u_nt:.2f}%")
            print(f"  NT-Shuffle u_nt:     {nt_stats['mean_u_nt']:.2f} +/- {nt_stats['std_u_nt']:.2f}%")
            print(f"  NT-Shuffle u_tr:     {nt_stats['mean_u_tr']:.2f} +/- {nt_stats['std_u_tr']:.2f}%")
            print(f"  NT-Shuffle u_non:    {nt_stats['mean_u_non']:.2f} +/- {nt_stats['std_u_non']:.2f}%")
            print(f"  p-value:             {nt_stats['p_value']:.4f}")
        else:
            print("  Failed to obtain any successful nt_shuffle realization.")

    # === Form unified result for full_analysis ===
    if NULL_MODEL_TYPE == 'full_analysis':
        null_stats = {
            'energy': energy_stats,
            'topo': topo_stats
        }
    elif NULL_MODEL_TYPE == 'energy_shuffle':
        null_stats = energy_stats
    elif NULL_MODEL_TYPE == 'topo_shuffle':
        null_stats = topo_stats
    elif NULL_MODEL_TYPE == 'nt_shuffle':
        null_stats = nt_stats

    return null_stats, control_stats


# ============================================================================
# SEQUENCE LOADING AND PROCESSING
# ============================================================================

def load_fasta_sequences():
    """
    Scans current folder for files with .fasta extension.
    """
    try:
        from Bio import SeqIO
    except ImportError:
        print("Error: biopython package required for FASTA file handling.")
        print("Install it with: pip install biopython")
        raise

    fasta_files = glob.glob("*.fasta")
    if not fasta_files:
        print("Error: no .fasta files found in current folder")
        return []

    print(f"Found FASTA files: {len(fasta_files)}")
    for f in fasta_files:
        print(f"  {f}")

    all_sequences = []
    for fasta_file in fasta_files:
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                seq_str = str(record.seq).upper().replace('T', 'U')
                filtered_seq = ''.join(c for c in seq_str if c in {'A', 'U', 'G', 'C'})
                if len(filtered_seq) >= 10:
                    all_sequences.append((filtered_seq, record.description if record.description else record.id, len(filtered_seq)))
        except Exception as e:
            print(f"  Error reading file {fasta_file}: {e}")

    all_sequences.sort(key=lambda x: x[2])

    print(f"\nLoaded sequences: {len(all_sequences)}")
    print("Sequences (sorted by increasing length):")
    for i, (seq, desc, length) in enumerate(all_sequences):
        print(f"  {i+1}. {desc}: length {length} nt")

    return all_sequences


def _single_stat_run(args):
    """
    Worker for parallel execution of one NUM_STAT run.
    Executes full pipeline: structure generation, graph, analysis of ALL components.
    Returns dictionary with results or None on error.
    """
    (run_idx, seq, seq_description, seq_len, current_seed, show_details) = args

    start_time = time.time()
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15

    np.random.seed(current_seed)

    comp_map = {('A', 'U'): True, ('U', 'A'): True, ('G', 'C'): True,
                ('C', 'G'): True, ('G', 'U'): True, ('U', 'G'): True}
    allowed_pairs, pair_to_idx, conflict_masks, bit, P = precompute_allowed_pairs_and_conflicts(
        seq_len, seq, MIN_HAIRPIN_LEN, comp_map
    )

    structures, energies = generate_structures_stochastic(
        seq, TEMPERATURE_CELSIUS, MAX_STRUCTURES, ENERGY_WINDOW, verbose=show_details
    )
    if len(structures) < 2:
        return None

    structures, energies = deduplicate_structures(structures, energies, verbose=show_details)

    struct_masks = []
    struct_set_bits = []
    index_map = {}
    for idx, s in enumerate(structures):
        mask, set_bits = dotbracket_to_bitmask(s, pair_to_idx)
        struct_masks.append(mask)
        struct_set_bits.append(set_bits)
        index_map[mask] = idx

    # Use local function for graph inside worker
    neighbors = _build_neighbor_graph_local(
        struct_masks, struct_set_bits, index_map, conflict_masks, bit, P
    )
    del struct_masks, struct_set_bits, index_map
    gc.collect()

    graph_components = find_connected_components(neighbors)
    n_total = len(energies)

    # Analysis of ALL significant components
    analysis = _analyze_all_components(
        energies, neighbors, graph_components, n_total,
        temp_kelvin, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, NUM_EIGENMODES,
        SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
        ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
        verbose=show_details
    )

    if analysis['total_triplets'] == 0:
        return None

    elapsed = time.time() - start_time

    return {
        'weighted_u_nt': analysis['weighted_u_nt'],
        'weighted_u_tr': analysis['weighted_u_tr'],
        'weighted_u_non': analysis['weighted_u_non'],
        'f_inter': analysis['f_inter'],
        'n_structures': n_total,
        'n_basins': analysis['total_basins'],
        'n_components': len(graph_components),
        'n_significant_components': analysis['n_significant_components'],
        'n_processed_components': analysis['n_processed_components'],
        'elapsed': elapsed,
        # Save full data for null tests (only from first successful run)
        'all_energies': energies,
        'all_neighbors': neighbors,
        'all_components': graph_components,
        'n_total_structures': n_total,
        'original_seq': seq,
        'global_basin_stats': analysis['global_basin_stats']
    }


def process_sequence(seq, seq_description, seq_index, total_sequences):
    """
    Performs full ultrametricity analysis for one RNA sequence.
    NUM_STAT runs are executed IN PARALLEL via multiprocessing.Pool.
    Null hypothesis statistical test is performed ONCE based on
    data from first successful run.
    """
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()

    print("\n" + "=" * 70)
    print(f"PROCESSING SEQUENCE {seq_index} OF {total_sequences}")
    print(f"Description: {seq_description}")
    print(f"Length: {seq_len} nucleotides")
    print(f"Seed: {RANDOM_SEED}")
    if NULL_MODEL_TYPE != 'none':
        print(f"NULL HYPOTHESIS TEST: {NULL_MODEL_TYPE} ({NUM_NULL_SAMPLES} realizations)")
    print("=" * 70)

    # === PARALLEL EXECUTION OF NUM_STAT RUNS ===
    main_start_time = time.time()

    worker_args = [
        (run_idx, seq, seq_description, seq_len, RANDOM_SEED + run_idx, VERBOSE)
        for run_idx in range(NUM_STAT)
    ]

    # Limit number of processes for main stage to avoid memory overflow
    n_stat_workers = min(n_workers, NUM_STAT)

    if NUM_STAT > 1:
        print(f"\nPARALLEL EXECUTION OF {NUM_STAT} RUNS ON {n_stat_workers} PROCESSES...")
        with Pool(processes=n_stat_workers) as pool:
            all_runs = pool.map(_single_stat_run, worker_args)
        # Filter failed runs
        all_runs = [r for r in all_runs if r is not None]
    else:
        # Single run -- execute directly
        result = _single_stat_run(worker_args[0])
        all_runs = [result] if result is not None else []

    main_elapsed = time.time() - main_start_time
    print(f"\n[LOG] Main stage time (NUM_STAT={NUM_STAT}): {main_elapsed:.2f} sec")

    if not all_runs:
        print("  ERROR: no runs completed successfully.")
        return None

    # === RESULT AGGREGATION ===
    weighted_u_nt_vals = np.array([r['weighted_u_nt'] for r in all_runs])
    weighted_u_tr_vals = np.array([r['weighted_u_tr'] for r in all_runs])
    weighted_u_non_vals = np.array([r['weighted_u_non'] for r in all_runs])
    f_inter_vals = np.array([r['f_inter'] for r in all_runs])
    n_structures_vals = np.array([r['n_structures'] for r in all_runs], dtype=np.float64)
    n_basins_vals = np.array([r['n_basins'] for r in all_runs], dtype=np.float64)
    n_components_vals = np.array([r['n_components'] for r in all_runs], dtype=np.float64)
    elapsed_vals = np.array([r['elapsed'] for r in all_runs])

    mean_u_nt = np.mean(weighted_u_nt_vals)
    std_u_nt = np.std(weighted_u_nt_vals, ddof=1) if len(all_runs) > 1 else 0.0
    mean_u_tr = np.mean(weighted_u_tr_vals)
    std_u_tr = np.std(weighted_u_tr_vals, ddof=1) if len(all_runs) > 1 else 0.0
    mean_u_non = np.mean(weighted_u_non_vals)
    std_u_non = np.std(weighted_u_non_vals, ddof=1) if len(all_runs) > 1 else 0.0
    f_inter = np.mean(f_inter_vals)
    f_inter_std = np.std(f_inter_vals, ddof=1) if len(all_runs) > 1 else 0.0
    mean_struct = int(round(np.mean(n_structures_vals)))
    std_struct = int(round(np.std(n_structures_vals, ddof=1))) if len(all_runs) > 1 else 0
    mean_basins = int(round(np.mean(n_basins_vals)))
    std_basins = int(round(np.std(n_basins_vals, ddof=1))) if len(all_runs) > 1 else 0
    mean_comp = int(round(np.mean(n_components_vals)))
    std_comp = int(round(np.std(n_components_vals, ddof=1))) if len(all_runs) > 1 else 0
    mean_elapsed = np.mean(elapsed_vals)
    std_elapsed = np.std(elapsed_vals, ddof=1) if len(all_runs) > 1 else 0.0

    first_res = all_runs[0]

    # === NULL HYPOTHESIS TEST EXECUTION (ONCE) ===
    null_stats = None
    control_stats = None

    null_start_time = time.time()

    if NULL_MODEL_TYPE != 'none':
        print("\n" + "-" * 50)
        print("RUNNING NULL HYPOTHESIS TEST (based on 1st successful run)")
        print("-" * 50)

        temp_kelvin = TEMPERATURE_CELSIUS + 273.15
        original_seq_for_test = first_res.get('original_seq', seq)

        # Form comp_res with full data for null tests
        comp_res_for_null = {
            'all_energies': first_res['all_energies'],
            'all_neighbors': first_res['all_neighbors'],
            'all_components': first_res['all_components'],
            'n_total_structures': first_res['n_total_structures'],
            # For random_basins need data of one component
            'basins': [],
            'comp_size': first_res['n_total_structures'],
            'eigenvalues_filtered': None,
            'eigenvectors_filtered': None
        }

        # For random_basins need basin and spectrum info of first component
        if NULL_MODEL_TYPE == 'random_basins':
            components = first_res['all_components']
            min_comp_size = max(3, int(ALPHA_COMPONENT_THRESHOLD * first_res['n_total_structures']))
            main_comp = None
            for comp in components:
                if len(comp) >= min_comp_size:
                    main_comp = comp
                    break

            if main_comp is not None:
                comp_set = set(main_comp)
                comp_energies = first_res['all_energies'][list(main_comp)]
                old_to_local = {old: local for local, old in enumerate(main_comp)}

                comp_neighbors = []
                for old_idx in main_comp:
                    local_nbs = {old_to_local[nb] for nb in first_res['all_neighbors'][old_idx] if nb in comp_set}
                    comp_neighbors.append(local_nbs)

                comp_basins_raw = compute_gradient_basins(comp_energies, comp_neighbors, verbose=False)
                RT = R_KCAL * temp_kelvin
                Z = {i: sum(np.exp(-comp_energies[idx] / RT) for idx in indices)
                     for i, (_, indices) in enumerate(comp_basins_raw)}
                filtered_basins, _ = filter_macrostates_spectral(
                    comp_basins_raw, Z, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, verbose=False
                )

                if len(filtered_basins) >= 3:
                    K_sym_comp = build_transition_rate_matrix(
                        comp_energies, comp_neighbors, temp_kelvin, FREQUENCY_PREFACTOR
                    )
                    num_requested = min(NUM_EIGENMODES, len(main_comp) - 1)
                    try:
                        _, evals, evecs, _, _ = compute_spectral_distance(
                            K_sym_comp, filtered_basins, num_requested, temp_kelvin,
                            SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA, verbose=False
                        )
                        comp_res_for_null['basins'] = filtered_basins
                        comp_res_for_null['comp_size'] = len(main_comp)
                        comp_res_for_null['eigenvalues_filtered'] = evals
                        comp_res_for_null['eigenvectors_filtered'] = evecs
                    except Exception:
                        pass

        null_stats, control_stats = run_null_hypothesis_tests(
            mean_u_nt,
            comp_res_for_null,
            temp_kelvin,
            n_workers,
            original_seq=original_seq_for_test,
            verbose=VERBOSE
        )

    null_elapsed = time.time() - null_start_time
    if NULL_MODEL_TYPE != 'none':
        print(f"\n[LOG] Null stage time ({NULL_MODEL_TYPE}): {null_elapsed:.2f} sec")

    # === STATISTICS OUTPUT ===
    print("\n" + "=" * 70)
    print(f"STATISTICS OVER {len(all_runs)} RUNS")
    print("=" * 70)
    print(f"  Weighted u_nt:        {mean_u_nt:.2f} +/- {std_u_nt:.2f} %")
    print(f"  Weighted u_tr:        {mean_u_tr:.2f} +/- {std_u_tr:.2f} %")
    print(f"  Weighted u_non:       {mean_u_non:.2f} +/- {std_u_non:.2f} %")
    print(f"  Inter-component frac: {f_inter:.4f} +/- {f_inter_std:.4f}")
    print(f"  Number of structures: {mean_struct} +/- {std_struct}")
    print(f"  Number of basins:     {mean_basins} +/- {std_basins}")
    print(f"  Components (total):   {mean_comp} +/- {std_comp}")
    print(f"  Execution time (main):{mean_elapsed:.1f} +/- {std_elapsed:.1f} sec")

    if null_stats:
        if NULL_MODEL_TYPE == 'full_analysis':
            print(f"  --- FULL MECHANISM ANALYSIS ---")
            if null_stats.get('topo'):
                ts = null_stats['topo']
                print(f"  TopoShuffle (chaos) u_nt:     {ts['mean_u_nt']:.2f} +/- {ts['std_u_nt']:.2f} %")
                print(f"  TopoShuffle (chaos) u_tr:     {ts['mean_u_tr']:.2f} +/- {ts['std_u_tr']:.2f} %")
                print(f"  TopoShuffle (chaos) u_non:    {ts['mean_u_non']:.2f} +/- {ts['std_u_non']:.2f} %")
                print(f"  TopoShuffle p-value:         {ts['p_value']:.4f}")
            if null_stats.get('energy'):
                es = null_stats['energy']
                print(f"  EnergyShuffle (topology) u_nt:  {es['mean_u_nt']:.2f} +/- {es['std_u_nt']:.2f} %")
                print(f"  EnergyShuffle (topology) u_tr:  {es['mean_u_tr']:.2f} +/- {es['std_u_tr']:.2f} %")
                print(f"  EnergyShuffle (topology) u_non: {es['mean_u_non']:.2f} +/- {es['std_u_non']:.2f} %")
                print(f"  EnergyShuffle p-value:           {es['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'energy_shuffle':
            print(f"  --- Test: energy shuffle ---")
            print(f"  Shuffled u_nt:          {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
            print(f"  Shuffled u_tr:          {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
            print(f"  Shuffled u_non:         {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
            print(f"  p-value:                {null_stats['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'topo_shuffle':
            print(f"  --- Test: configuration model ---")
            print(f"  Chaotic u_nt:           {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
            print(f"  Chaotic u_tr:           {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
            print(f"  Chaotic u_non:          {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
            print(f"  p-value:                {null_stats['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'nt_shuffle':
            print(f"  --- Test: nucleotide shuffle ---")
            if null_stats:
                print(f"  NT-Shuffle u_nt:            {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
                print(f"  NT-Shuffle u_tr:            {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
                print(f"  NT-Shuffle u_non:           {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
                print(f"  p-value:                    {null_stats['p_value']:.4f}")
            else:
                print("  Results unavailable (see errors above)")

    if control_stats:
        print(f"  --- Control test (random basins) ---")
        print(f"  Random u_nt:             {control_stats['mean_u_nt']:.2f} +/- {control_stats['std_u_nt']:.2f} %")
        print(f"  p-value:                 {control_stats['p_value']:.4f}")

    # Form final dictionary
    result_dict = {
        'sequence': seq[:50], 'description': seq_description, 'length': seq_len,
        'weighted_u_nt': mean_u_nt, 'weighted_u_nt_std': std_u_nt,
        'weighted_u_tr': mean_u_tr, 'weighted_u_tr_std': std_u_tr,
        'weighted_u_non': mean_u_non, 'weighted_u_non_std': std_u_non,
        'f_inter': f_inter, 'f_inter_std': f_inter_std,
        'n_structures': mean_struct, 'n_structures_std': std_struct,
        'n_basins': mean_basins, 'n_basins_std': std_basins,
        'n_components': mean_comp, 'n_components_std': std_comp,
        'n_significant_components': np.mean([r['n_significant_components'] for r in all_runs]),
        'n_processed_components': first_res['n_processed_components'],
        'method': 'spectral_mahalanobis',
        'elapsed': mean_elapsed, 'elapsed_std': std_elapsed,
        'num_runs': len(all_runs), 'all_runs': all_runs,
        'null_model_type': NULL_MODEL_TYPE
    }

    # Save null test results
    if NULL_MODEL_TYPE == 'full_analysis' and null_stats:
        if null_stats.get('energy'):
            es = null_stats['energy']
            result_dict.update({
                'energy_p_value': es['p_value'],
                'energy_u_nt_mean': es['mean_u_nt'], 'energy_u_nt_std': es['std_u_nt'],
                'energy_u_tr_mean': es['mean_u_tr'], 'energy_u_tr_std': es['std_u_tr'],
                'energy_u_non_mean': es['mean_u_non'], 'energy_u_non_std': es['std_u_non'],
                'energy_elapsed': es['elapsed']
            })
        if null_stats.get('topo'):
            ts = null_stats['topo']
            result_dict.update({
                'topo_p_value': ts['p_value'],
                'topo_u_nt_mean': ts['mean_u_nt'], 'topo_u_nt_std': ts['std_u_nt'],
                'topo_u_tr_mean': ts['mean_u_tr'], 'topo_u_tr_std': ts['std_u_tr'],
                'topo_u_non_mean': ts['mean_u_non'], 'topo_u_non_std': ts['std_u_non'],
                'topo_elapsed': ts['elapsed']
            })
    elif null_stats and NULL_MODEL_TYPE in ('energy_shuffle', 'topo_shuffle', 'nt_shuffle'):
        prefix = 'null'
        result_dict.update({
            f'{prefix}_p_value_mean': null_stats['p_value'],
            f'{prefix}_u_nt_mean': null_stats['mean_u_nt'], f'{prefix}_u_nt_std': null_stats['std_u_nt'],
            f'{prefix}_u_tr_mean': null_stats['mean_u_tr'], f'{prefix}_u_tr_std': null_stats['std_u_tr'],
            f'{prefix}_u_non_mean': null_stats['mean_u_non'], f'{prefix}_u_non_std': null_stats['std_u_non'],
            f'{prefix}_elapsed': null_stats['elapsed']
        })

    if control_stats:
        result_dict.update({
            'control_p_value_mean': control_stats['p_value'],
            'control_u_nt_mean': control_stats['mean_u_nt'],
            'control_elapsed': control_stats['elapsed']
        })

    return result_dict


def main():
    """
    Main entry point. Manages data loading, processing launch,
    and formation of final summary report.
    """
    total_start_time = time.time()

    # === OUTPUT ALL COMPUTATION PARAMETERS ===
    print("=" * 70)
    print("COMPUTATION PARAMETERS")
    print("=" * 70)
    print(f"  FASTA_RNA:                {FASTA_RNA}")
    print(f"  TEMPERATURE_CELSIUS:      {TEMPERATURE_CELSIUS}")
    print(f"  ENERGY_WINDOW:            {ENERGY_WINDOW}")
    print(f"  MAX_STRUCTURES:           {MAX_STRUCTURES}")
    print(f"  MIN_HAIRPIN_LEN:          {MIN_HAIRPIN_LEN}")
    print(f"  RANDOM_SEED:              {RANDOM_SEED}")
    print(f"  MAX_MACROSTATES_ANALYSIS: {MAX_MACROSTATES_ANALYSIS}")
    print(f"  MIN_MACROSTATE_SIZE:      {MIN_MACROSTATE_SIZE}")
    print(f"  ALPHA_COMPONENT_THRESHOLD:{ALPHA_COMPONENT_THRESHOLD}")
    print(f"  NUM_EIGENMODES:           {NUM_EIGENMODES}")
    print(f"  SPECTRAL_GAP_THRESHOLD:   {SPECTRAL_GAP_THRESHOLD}")
    print(f"  FREQUENCY_PREFACTOR:      {FREQUENCY_PREFACTOR}")
    print(f"  EIGS_MAXITER:             {EIGS_MAXITER}")
    print(f"  EIGS_SIGMA:               {EIGS_SIGMA}")
    print(f"  ULTRAMETRIC_EPSILON:      {ULTRAMETRIC_EPSILON}")
    print(f"  ULTRAMETRIC_DELTA:        {ULTRAMETRIC_DELTA}")
    print(f"  EPS_COMPARISON:           {EPS_COMPARISON}")
    print(f"  NUM_WORKERS:              {NUM_WORKERS}")
    print(f"  VERBOSE:                  {VERBOSE}")
    print(f"  NUM_STAT:                 {NUM_STAT}")
    print(f"  NULL_MODEL_TYPE:          {NULL_MODEL_TYPE}")
    print(f"  NUM_NULL_SAMPLES:         {NUM_NULL_SAMPLES}")
    print(f"  NUM_EDGE_SWAPS_MULTIPLIER:{NUM_EDGE_SWAPS_MULTIPLIER}")
    print(f"  EXPECTATION_BY_RNA:       {EXPECTATION_BY_RNA}")
    print("=" * 70)

    # Resource intensity warning
    heavy_modes = ('energy_shuffle', 'topo_shuffle', 'full_analysis', 'nt_shuffle')
    if NULL_MODEL_TYPE in heavy_modes and NUM_NULL_SAMPLES > 10:
        print("=" * 70)
        print(f"WARNING: You selected '{NULL_MODEL_TYPE}' mode with large number of realizations.")
        print(f"Number of realizations: {NUM_NULL_SAMPLES}")
        if NULL_MODEL_TYPE == 'nt_shuffle':
            print("nt_shuffle mode requires FULL recalculation for each realization.")
            print("This is VERY SLOW. Recommended NUM_NULL_SAMPLES <= 5.")
        elif NULL_MODEL_TYPE == 'full_analysis':
            print("full_analysis mode performs TWO heavy tests sequentially.")
            print("Recommended to reduce NUM_NULL_SAMPLES to 20.")
        elif NULL_MODEL_TYPE == 'topo_shuffle':
            print("Each realization requires graph rewiring, basin and spectrum recalculation.")
            print("Recommended to reduce NUM_NULL_SAMPLES to 20.")
        else:
            print("Each realization requires full transition matrix spectrum recalculation.")
            print("Recommended to reduce NUM_NULL_SAMPLES to 20-30.")
        print("=" * 70)
        time.sleep(5)

    print("=" * 70)
    print("CALCULATION OF NONTRIVIAL ULTRAMETRICITY DEGREE")
    print("FOR SECONDARY STRUCTURE RNA MACROSTATES")
    print("METHOD: spectral Mahalanobis distance")
    print("(physically rigorous approach via transition rate matrix)")
    print("STRUCTURE GENERATION: stochastic sampling (pbacktrack)")
    if NUM_STAT > 1: print(f"STATISTICAL MODE: {NUM_STAT} runs (PARALLEL)")
    if NULL_MODEL_TYPE != 'none': print(f"NULL HYPOTHESIS TEST: {NULL_MODEL_TYPE} ({NUM_NULL_SAMPLES} realizations, PARALLEL)")
    if NULL_MODEL_TYPE == 'none' and EXPECTATION_BY_RNA: print(f"RNA ENSEMBLE EXPECTATION: ENABLED")
    print("=" * 70)

    sequences = load_fasta_sequences() if FASTA_RNA else [(RNA_SEQUENCE, "RNA_SEQUENCE (from parameter)", len(RNA_SEQUENCE))]
    if not sequences: return

    all_seq_results = []
    for seq_idx, (seq, seq_desc, seq_len) in enumerate(sequences, start=1):
        result = process_sequence(seq, seq_desc, seq_idx, len(sequences))
        if result is not None: all_seq_results.append(result)
        gc.collect()

    if len(sequences) > 1 and all_seq_results:
        is_full = NULL_MODEL_TYPE == 'full_analysis'
        has_energy = NULL_MODEL_TYPE == 'energy_shuffle' or is_full
        has_topo = NULL_MODEL_TYPE == 'topo_shuffle' or is_full
        has_nt = NULL_MODEL_TYPE == 'nt_shuffle'
        has_control = NULL_MODEL_TYPE == 'random_basins'
        has_any_test = has_energy or has_topo or has_nt or has_control

        # === TABLE 1: MAIN RESULTS ===
        print("\n" + "=" * 70)
        print("FINAL SUMMARY REPORT: MAIN RESULTS")
        print("=" * 70)

        header_main = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Structures':<16} {'Basins':<16} {'f_inter':<14} {'u_nt (%)':<18} {'u_tr (%)':<18} {'u_non (%)':<18} {'Time (s)':<16}"
        print(header_main)
        print("-" * len(header_main))

        for i, res in enumerate(all_seq_results):
            desc = res['description'][:28]
            if NUM_STAT > 1:
                line = (f"{i+1:<4} {desc:<30} {res['length']:<8} "
                        f"{int(res['n_structures'])}+/-{int(res['n_structures_std']):<14} "
                        f"{int(res['n_basins'])}+/-{int(res['n_basins_std']):<14} "
                        f"{res['f_inter']:.4f}+/-{res['f_inter_std']:.4f}{'':3} "
                        f"{res['weighted_u_nt']:.2f}+/-{res['weighted_u_nt_std']:.2f}{'':3} "
                        f"{res['weighted_u_tr']:.2f}+/-{res['weighted_u_tr_std']:.2f}{'':3} "
                        f"{res['weighted_u_non']:.2f}+/-{res['weighted_u_non_std']:.2f}{'':3} "
                        f"{res['elapsed']:.1f}+/-{res['elapsed_std']:.1f}")
            else:
                line = (f"{i+1:<4} {desc:<30} {res['length']:<8} {res['n_structures']:<16} {res['n_basins']:<16} "
                        f"{res['f_inter']:.4f}{'':9} {res['weighted_u_nt']:.2f}{'':13} {res['weighted_u_tr']:.2f}{'':13} "
                        f"{res['weighted_u_non']:.2f}{'':13} {res['elapsed']:<16.1f}")
            print(line)

        # === NULL HYPOTHESIS TEST RESULT TABLES ===
        if has_any_test:
            if has_energy:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: ENERGY SHUFFLE TEST (TOPOLOGY CONTRIBUTION)")
                print("=" * 70)

                header_e = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'Energy u_nt':<16} {'Energy u_tr':<16} {'Energy u_non':<16} {'p(Energy)':<10} {'Time (s)':<10}"
                print(header_e)
                print("-" * len(header_e))

                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"

                    e_nt = res.get('energy_u_nt_mean')
                    e_tr = res.get('energy_u_tr_mean')
                    e_non = res.get('energy_u_non_mean')
                    e_pv = res.get('energy_p_value')
                    e_el = res.get('energy_elapsed')

                    e_nt_str = (f"{e_nt:.2f}+/-{res['energy_u_nt_std']:.2f}" if e_nt is not None else "N/A")
                    e_tr_str = (f"{e_tr:.2f}+/-{res['energy_u_tr_std']:.2f}" if e_tr is not None else "N/A")
                    e_non_str = (f"{e_non:.2f}+/-{res['energy_u_non_std']:.2f}" if e_non is not None else "N/A")
                    e_pv_str = (f"{e_pv:.4f}" if e_pv is not None else "N/A")
                    e_el_str = (f"{e_el:.1f}" if e_el is not None else "N/A")

                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {e_nt_str:<16} {e_tr_str:<16} {e_non_str:<16} {e_pv_str:<10} {e_el_str:<10}"
                    print(line)

            if has_topo:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: TOPO SHUFFLE TEST (CONFIGURATION MODEL)")
                print("=" * 70)

                header_t = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'Topo u_nt':<16} {'Topo u_tr':<16} {'Topo u_non':<16} {'p(Topo)':<10} {'Time (s)':<10}"
                print(header_t)
                print("-" * len(header_t))

                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"

                    t_nt = res.get('topo_u_nt_mean')
                    t_tr = res.get('topo_u_tr_mean')
                    t_non = res.get('topo_u_non_mean')
                    t_pv = res.get('topo_p_value')
                    t_el = res.get('topo_elapsed')

                    t_nt_str = (f"{t_nt:.2f}+/-{res['topo_u_nt_std']:.2f}" if t_nt is not None else "N/A")
                    t_tr_str = (f"{t_tr:.2f}+/-{res['topo_u_tr_std']:.2f}" if t_tr is not None else "N/A")
                    t_non_str = (f"{t_non:.2f}+/-{res['topo_u_non_std']:.2f}" if t_non is not None else "N/A")
                    t_pv_str = (f"{t_pv:.4f}" if t_pv is not None else "N/A")
                    t_el_str = (f"{t_el:.1f}" if t_el is not None else "N/A")

                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {t_nt_str:<16} {t_tr_str:<16} {t_non_str:<16} {t_pv_str:<10} {t_el_str:<10}"
                    print(line)

            if has_nt:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: NT SHUFFLE TEST (NUCLEOTIDE SHUFFLING)")
                print("=" * 70)

                header_nt = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'NT u_nt':<16} {'NT u_tr':<16} {'NT u_non':<16} {'p(NT)':<10} {'Time (s)':<10}"
                print(header_nt)
                print("-" * len(header_nt))

                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"

                    n_nt = res.get('null_u_nt_mean')
                    n_tr = res.get('null_u_tr_mean')
                    n_non = res.get('null_u_non_mean')
                    n_pv = res.get('null_p_value_mean')
                    n_el = res.get('null_elapsed')

                    n_nt_str = (f"{n_nt:.2f}+/-{res['null_u_nt_std']:.2f}" if n_nt is not None else "N/A")
                    n_tr_str = (f"{n_tr:.2f}+/-{res['null_u_tr_std']:.2f}" if n_tr is not None else "N/A")
                    n_non_str = (f"{n_non:.2f}+/-{res['null_u_non_std']:.2f}" if n_non is not None else "N/A")
                    n_pv_str = (f"{n_pv:.4f}" if n_pv is not None else "N/A")
                    n_el_str = (f"{n_el:.1f}" if n_el is not None else "N/A")

                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {n_nt_str:<16} {n_tr_str:<16} {n_non_str:<16} {n_pv_str:<10} {n_el_str:<10}"
                    print(line)

    print(f"\nTotal execution time: {time.time() - total_start_time:.1f} seconds")


if __name__ == "__main__":
    main()

