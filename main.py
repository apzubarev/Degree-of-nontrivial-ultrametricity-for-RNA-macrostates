"""
Calculation of the degree of nontrivial ultrametricity for RNA macrostates.
PHYSICALLY RIGOROUS APPROACH: distance between basins via spectral
decomposition of the transition rate matrix (Mahalanobis distance
in the space of eigenvectors of the symmetrized matrix K).

METHOD:
1. A transition rate matrix K is built between all structures
   (N x N, where N ~ 2000) based on the Kramers formula.
2. K is symmetrized taking detailed balance into account.
3. The m smallest eigenvalues in magnitude and corresponding
   eigenvectors are computed (Lanczos method for sparse matrices).
4. Automatic filtering of noise modes is performed by finding
   a spectral gap: if the ratio |λ_k| / |λ_{k-1}| exceeds
   a threshold (default 10^6), modes with indices < k are discarded
   as numerical noise.
5. Each attraction basin is represented by a characteristic
   vector χ_A in the space of structures.
6. The distance between basins A and B is defined as the weighted
   Euclidean distance between projections of χ_A and χ_B onto
   eigenvectors (Mahalanobis distance).
7. The resulting distance matrix is a metric and is tested
   for ultrametricity.

HANDLING DISCONNECTED GRAPHS:
Before constructing the K_sym matrix, the connectivity of the structure
graph is checked. If the graph contains multiple connected components,
each component is processed separately: its own K_sym matrix is built,
spectral decomposition is performed, and ultrametricity is checked.
Components with fewer than 3 basins are skipped.

STATISTICAL MODE (NUM_STAT > 1):
When NUM_STAT > 1, NUM_STAT independent runs are performed for each
sequence with different random samples of structures (seed varies:
RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
Results are averaged, and the final table shows mean values and
standard deviations (mean ± std). Integer quantities (number of
structures, basins, connected components) are rounded to integers.

OUTPUT MODES:
VERBOSE = True  — full log (steps, components, spectral analysis).
VERBOSE = False — brief log: sequence header and parameters are
                  printed once, then only RUN/COMPLETED, followed
                  by a statistics block.

ADVANTAGES:
- Takes into account all possible transition paths (via spectral decomposition).
- Context-independent (distance between A and B is determined only
  by them, not by the presence of other basins).
- Symmetric and guaranteed to be a metric.
- Automatically filters out numerical noise via spectral gap detection.
- Correctly handles disconnected structure graphs.
- Computational complexity O(m·N·E + K²·m), allowing processing
  of N ~ 2000 structures and K ~ 100 basins in seconds.

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
  True  — scan current folder for *.fasta files,
          load all sequences, sort by length.
  False — use sequence from RNA_SEQUENCE.
Recommended value: True (for research work).
"""

RNA_SEQUENCE = "ACATCAATCCACCACTCTTTCTCTTTAAAAAGAGTAGACCCAGGAACCGAAATTCTTTACCAAATTAAAAAA"
"""
Primary RNA structure. Used only when FASTA_RNA = False.
Allowed characters: A, U, G, C (uppercase, T is automatically replaced by U).
Recommended length: 50–200 nucleotides.
"""

# --- Temperature and energy parameters ---

TEMPERATURE_CELSIUS = 37.0
"""
Temperature in degrees Celsius.
Affects Boltzmann weights and transition probabilities.
  Low (< 20°C): deep basins, rare transitions.
  High (> 60°C): smoothed landscape, fast transitions.
Recommended value: 37.0 (physiological temperature).
Valid range: 0.0 – 100.0.
"""

ENERGY_WINDOW = 100.0
"""
Energy window (kcal/mol) relative to minimum free energy (MFE).
During stochastic sampling, structures with energy > MFE + ENERGY_WINDOW
are discarded. If set to "inf", there is no window limit.
  Small window (1–5 kcal/mol): only most stable structures,
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
Generation stops when the number of unique structures within the
specified energy window reaches this value.
  Few (100–500): fast but statistically poor analysis.
  Many (> 10000): complete landscape picture, but slow.
Recommended value: 2000–5000.
Valid range: 100 – 20000.
"""

MIN_HAIRPIN_LEN = 3
"""
Minimum number of unpaired nucleotides in a hairpin loop.
Defines condition: j - i - 1 >= MIN_HAIRPIN_LEN.
  Standard value: 3 (steric constraint).
  Value 0 disables constraint (unphysical).
Recommended value: 3.
Valid range: 0 – 10.
"""

RANDOM_SEED = 43
"""
Initial seed for random number generator.
Ensures reproducibility of results.
When NUM_STAT > 1, seed varies: RANDOM_SEED, RANDOM_SEED+1, ...
Recommended value: 42 (or any integer).
Valid range: any integer.
"""

# --- Attraction basin parameters ---

MAX_MACROSTATES_ANALYSIS = 500
"""
Maximum number of attraction basins included in final analysis.
If more remain after filtering, basins with largest partition
functions Z are kept.
  Few (10–30): fast, but may lack triplet statistics.
  Many (> 200): more triplets for analysis, but slower (K³ for spectrum).
Recommended value: 100.
Valid range: 3 – 500.
"""

MIN_MACROSTATE_SIZE = 50
"""
Minimum size of attraction basin (number of constituent structures).
Smaller basins are considered statistically insignificant.
  Value 1: includes all basins, including isolated structures.
  Value 5–10: filters out small artifactual basins.
Recommended value: 5.
Valid range: 1 – 100.
"""

# --- Spectral analysis parameters ---

NUM_EIGENMODES = 50
"""
Number of eigenmodes (eigenvalues and eigenvectors) requested
for spectral decomposition. After automatic noise mode filtering,
actual number of used modes may be smaller.
  Few (5–10): fast, but loses fine landscape structure info.
  Many (> 100): more accurate, but slower (scales linearly).
  Constraint: must be strictly less than number of structures.
Recommended value: 50.
Valid range: 5 – 200 (but no more than N-2, where N is number of structures).
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for detecting spectral gap between noise and physical modes.
If ratio |λ_k| / |λ_{k-1}| > SPECTRAL_GAP_THRESHOLD, modes with indices
< k are considered numerical noise and discarded.
  High threshold (10^8): conservative, may lose weak physical modes.
  Low threshold (10^2): aggressive, may retain noise modes.
Recommended value: 1e6.
Valid range: 1e2 – 1e12.
"""

FREQUENCY_PREFACTOR = 1.0
"""
Frequency prefactor ν₀ in Kramers formula (arbitrary units).
Affects absolute scale of matrix K, but does not affect eigenvectors
or relative distances between basins (changing ν₀ multiplies all λ_k
by a constant, which cancels in Mahalanobis distance).
Recommended value: 1.0 (leave unchanged).
Valid range: any positive number.
"""

EIGS_MAXITER = 50000
"""
Maximum iterations for Lanczos algorithm (ARPACK) when computing
eigenvalues of K_sym. Increasing this improves convergence for
matrices with dense spectrum near zero, but increases runtime.
Recommended value: 50000.
Valid range: 1000 – 200000.
"""

EIGS_SIGMA = 1e-10
"""
Shift sigma for Lanczos algorithm when searching for eigenvalues
near zero. Should be positive and small enough not to distort
physical mode spectrum (which have |λ| >= 10^-4), but large
enough to avoid numerical singularity when solving (K_sym - sigma*I)x = b.
  Too small (10^-15): risk of numerical singularity.
  Too large (10^-3): distorts spectrum.
Recommended value: 1e-10.
Valid range: 1e-12 – 1e-6.
"""

# --- Ultrametricity check parameters ---

ULTRAMETRIC_EPSILON = 0.05
"""
Relative tolerance ε for approximate ultrametricity check.
Two largest triangle sides are considered equal if
(d_max - d_mid) / d_mid <= ε.
  Must be strictly less than ULTRAMETRIC_DELTA.
  At ε = 0: exact equality required (almost unattainable).
  At ε > 0.1: many false positive classifications.
Recommended value: 0.05.
Valid range: 0.0 – 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference δ between smaller and middle triangle
sides for classification as nontrivially ultrametric:
(d_mid - d_min) / d_mid > δ.
  Must be strictly greater than ULTRAMETRIC_EPSILON.
  At small δ: equilateral triangles mistakenly classified
    as nontrivially ultrametric.
  At large δ: almost no nontrivially ultrametric triplets remain.
Recommended value: 0.1.
Valid range: 0.01 – 0.50.
"""

# --- Computational resource parameters ---

NUM_WORKERS = None
"""
Number of parallel processes for neighbor structure generation.
  None: automatically use all available CPU cores.
  1: single-threaded mode (for debugging).
  N: use exactly N processes.
Recommended value: None.
Valid range: 1 – cpu_count().
"""

VERBOSE = False
"""
Verbose output mode.
  True: print all intermediate results (basin sizes,
    transition stats, triangle distributions).
  False: only final results (brief log).
Recommended value: True (for research purposes).
"""

# --- Statistical analysis parameter ---

NUM_STAT = 5
"""
Number of statistical trials (independent runs) for each
RNA sequence.
  NUM_STAT = 1: single run, result without deviation.
  NUM_STAT > 1: performs NUM_STAT runs with different seeds
    (RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
    Results are averaged, output as mean ± SD.
    Integer quantities (number of structures, basins, components)
    are rounded to integers.
Recommended value: 1.
Valid range: 1 – 100.
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

R_KCAL = 0.001987204259  # Gas constant in kcal/(mol·K) (R = N_A * k_B)
EPS_COMPARISON = 1e-9  # For floating point comparison


# ============================================================================
# OPTIMIZATION: BITMASKS AND PRECOMPUTATION OF ALLOWED PAIRS + CONFLICTS
# ============================================================================

def precompute_allowed_pairs_and_conflicts(seq_len, sequence, min_hairpin_len, comp_map):
    """
    Precomputes list of all allowed pairs and conflict matrix between them.
    Conflicts are encoded as bitmasks for O(1) checking.
    
    Returns:
        allowed (list): list of pairs (i, j)
        pair_to_idx (dict): mapping from pair to its index
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
    
    new_structures = []
    new_energies = []
    for s, e in unique_pairs.values():
        new_structures.append(s)
        new_energies.append(e)
    
    if verbose and len(new_structures) < len(structures):
        print(f"  Duplicates removed: {len(structures) - len(new_structures)}")
    
    return new_structures, np.array(new_energies)


# ============================================================================
# OPTIMIZATION: PARALLEL NEIGHBOR GENERATION (BITMASK IPC)
# ============================================================================

def _pool_initializer_bitmask(index_map, conflict_masks, bit, P):
    """
    Initializer for Pool processes.
    Sets module-level globals for O(1) access.
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
    This drastically reduces IPC overhead.
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


# ============================================================================
# GRAPH BUILDING AND ANALYSIS FUNCTIONS
# ============================================================================

def generate_structures_stochastic(seq, temp_celsius, max_structures, energy_window, verbose=True):
    """
    Generates set of secondary structures by stochastic sampling
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
    total_generated = 0
    total_rejected_energy = 0
    
    for batch in range(max_batches):
        if len(structures) >= max_structures:
            break
        try:
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
            print(f"  Structures rejected above energy threshold ({energy_cutoff:.2f} kcal/mol): {total_rejected_energy}")
        print(f"  Stochastic sampling: generated {len(structures)} unique structures "
              f"(requested {max_structures})")
        if len(structures) < max_structures:
            print(f"  Warning: could not generate requested number of structures within energy window")
    
    return structures, np.array(energies_list)


def build_neighbor_graph_bitmask(struct_masks, struct_set_bits, index_map, conflict_masks, bit, P, num_workers=None, verbose=True):
    """
    Builds neighbor graph using bitmask-based neighbor generation.
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
    Finds connected components of structure graph.
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


def find_local_minima(energies, neighbors_list):
    """
    Finds all local minima in neighbor graph.
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
    with correct plateau handling.
    """
    n = len(energies)
    
    candidate_set = set()
    for i in range(n):
        has_lower = False
        for nb in neighbors_list[i]:
            if energies[nb] < energies[i] - EPS_COMPARISON:
                has_lower = True
                break
        if not has_lower:
            candidate_set.add(i)
    
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
            print(f"    Of which plateaus (size > 1): {plateau_count}")
            print(f"    Plateau sizes: {plateau_sizes}")
    
    basin_of = [-1] * n
    
    def find_basin(i):
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
# TRANSITION RATE MATRIX K CONSTRUCTION (STRUCTURE LEVEL)
# ============================================================================

def build_transition_rate_matrix(energies, neighbors_list, temp_kelvin, nu0):
    """
    Builds symmetrized transition rate matrix K_sym.
    """
    N = len(energies)
    RT = R_KCAL * temp_kelvin
    
    K_sym = lil_matrix((N, N), dtype=np.float64)
    row_sums = np.zeros(N, dtype=np.float64)
    
    for p in range(N):
        G_p = energies[p]
        
        for q in neighbors_list[p]:
            if q > p:
                G_q = energies[q]
                delta_G = abs(G_p - G_q)
                rate = nu0 * np.exp(-delta_G / (2.0 * RT))
                
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
    
    filtered_vals = sorted_vals[num_noise:]
    filtered_vecs = sorted_vecs[:, num_noise:]
    
    return filtered_vals, filtered_vecs, num_noise


def filter_macrostates_spectral(basins, Z, min_size, max_macrostates, verbose=True):
    """
    Filters macrostates by size and statistical significance.
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
            print(f"  Macrostates kept with largest Z: {len(valid)} (out of {len(basins)})")
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
    Computes Mahalanobis distance between basins.
    """
    N = K_sym.shape[0]
    K_basins = len(basins)
    
    max_possible = N - 1
    if num_modes_requested > max_possible:
        if verbose:
            print(f"  Warning: requested modes ({num_modes_requested}) > N-1 ({max_possible})")
        num_modes_requested = max_possible
    
    ncv = min(2 * num_modes_requested + 10, N)
    
    if verbose:
        print(f"  Computing {num_modes_requested} eigenmodes of {N}x{N} matrix...")
    
    eigenvalues = None
    eigenvectors = None
    last_error = None
    
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
            print(f"    Success (SM method, {eigs_maxiter} iterations)")
    except Exception as e:
        last_error = e
        if verbose:
            print(f"    SM method failed: {e}")
    
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
                print(f"    Success (LM method with shift sigma={eigs_sigma})")
        except Exception as e:
            last_error = e
            if verbose:
                print(f"    LM method with shift failed: {e}")
    
    if eigenvalues is None:
        raise RuntimeError(
            f"Failed to compute eigenvalues after two attempts. "
            f"Last error: {last_error}"
        )
    
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
            "All eigenmodes filtered as noise. "
            "Check SPECTRAL_GAP_THRESHOLD and temperature parameters."
        )
    
    if verbose:
        print(f"  Building characteristic vectors for {K_basins} basins...")
    
    chi = np.zeros((K_basins, N), dtype=np.float64)
    for a, (_, indices) in enumerate(basins):
        norm = np.sqrt(len(indices))
        chi[a, indices] = 1.0 / norm
    
    proj = chi @ eigenvectors_filtered
    weights = 1.0 / np.abs(eigenvalues_filtered)
    
    if verbose:
        print(f"  Computing {K_basins}x{K_basins} distance matrix...")
    
    dist_matrix = np.zeros((K_basins, K_basins), dtype=np.float64)
    
    for a in range(K_basins):
        for b in range(a + 1, K_basins):
            diff = proj[a, :] - proj[b, :]
            d_sq = np.sum(weights * (diff ** 2))
            dist_matrix[a, b] = np.sqrt(d_sq)
            dist_matrix[b, a] = dist_matrix[a, b]
    
    dist_matrix *= R_KCAL * temp_kelvin
    
    return dist_matrix, eigenvalues_filtered, eigenvectors_filtered, num_noise


# ============================================================================
# ULTRAMETRICITY CHECK
# ============================================================================

def classify_triangle(d1, d2, d3, eps, delta):
    """
    Classifies triangle by ultrametricity.
    """
    if d1 == float('inf') or d2 == float('inf') or d3 == float('inf'):
        return 'non_ultrametric'
    
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
    """Computes ultrametricity scores."""
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
# FASTA SEQUENCE LOADING FUNCTION
# ============================================================================

def load_fasta_sequences():
    """
    Scans current folder for .fasta files.
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
                    print(f"  Warning: invalid characters found in sequence {desc_full}, skipped.")
                if len(filtered_seq) < 10:
                    print(f"  Warning: sequence {desc_full} too short (< 10 nt), skipping.")
                    continue
                all_sequences.append((filtered_seq, desc_full, len(filtered_seq)))
        except Exception as e:
            print(f"  Error reading file {fasta_file}: {e}")
    
    if not all_sequences:
        print("Error: failed to load any sequences from FASTA files")
        return []
    
    all_sequences.sort(key=lambda x: x[2])
    
    print(f"\nSequences loaded: {len(all_sequences)}")
    print("Sequences (sorted by ascending length):")
    for i, (seq, desc, length) in enumerate(all_sequences):
        print(f"  {i+1}. {desc}: length {length} nt")
    
    return all_sequences


# ============================================================================
# SINGLE SEQUENCE PROCESSING FUNCTION (ONE RUN)
# ============================================================================

def process_sequence_single(seq, seq_description, seq_index, total_sequences, 
                            stat_iter, num_stat, current_seed, show_details=True):
    """
    Performs full ultrametricity analysis for one RNA sequence.
    """
    start_time = time.time()
    step_timings = {}
    
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15
    RT = R_KCAL * temp_kelvin
    
    np.random.seed(current_seed)
    
    # ===== STEP 1: PRECOMPUTE ALLOWED PAIRS AND CONFLICTS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 1: PRECOMPUTE ALLOWED PAIRS AND CONFLICTS")
        print("-" * 50)
    
    comp_map = {
        ('A', 'U'): True, ('U', 'A'): True,
        ('G', 'C'): True, ('C', 'G'): True,
        ('G', 'U'): True, ('U', 'G'): True,
    }
    allowed_pairs, pair_to_idx, conflict_masks, bit, P = precompute_allowed_pairs_and_conflicts(
        seq_len, seq, MIN_HAIRPIN_LEN, comp_map
    )
    if show_details:
        print(f"  Total allowed pairs: {P}")
    step_timings['Step 1: Precompute pairs and conflicts'] = time.time() - step_start
    
    # ===== STEP 2: GENERATE STRUCTURES =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 2: GENERATE SECONDARY STRUCTURES (stochastic sampling, pbacktrack)")
        print("-" * 50)
    
    structures, energies = generate_structures_stochastic(
        seq, TEMPERATURE_CELSIUS, MAX_STRUCTURES, ENERGY_WINDOW, verbose=show_details
    )
    
    if show_details:
        print(f"Structures generated: {len(structures)}")
    step_timings['Step 2: Generate structures'] = time.time() - step_start
    
    if len(structures) < 2:
        print("Error: insufficient structures for analysis")
        gc.collect()
        return None
    
    # ===== STEP 3: REMOVE DUPLICATES =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 3: REMOVE DUPLICATES")
        print("-" * 50)
    
    structures, energies = deduplicate_structures(structures, energies, verbose=show_details)
    if show_details:
        print(f"Unique structures: {len(structures)}")
    step_timings['Step 3: Remove duplicates'] = time.time() - step_start
    
    # ===== STEP 4: CONVERT TO BITMASKS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 4: CONVERT TO BITMASKS AND INDEX")
        print("-" * 50)
    
    struct_masks = []
    struct_set_bits = []
    index_map = {}
    for idx, s in enumerate(structures):
        mask, set_bits = dotbracket_to_bitmask(s, pair_to_idx)
        struct_masks.append(mask)
        struct_set_bits.append(set_bits)
        index_map[mask] = idx
        
    if show_details:
        print(f"Unique structures (indexed): {len(index_map)}")
    step_timings['Step 4: Convert to masks'] = time.time() - step_start
    
    # ===== STEP 5: BUILD NEIGHBOR GRAPH =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 5: BUILD NEIGHBOR GRAPH (Optimized IPC and Bitmask)")
        print("-" * 50)
        print("  Parallel generation with global variables and streaming")
    
    neighbors = build_neighbor_graph_bitmask(
        struct_masks, struct_set_bits, index_map, conflict_masks, bit, P, n_workers, verbose=show_details
    )
    edges = sum(len(nb) for nb in neighbors) // 2
    if show_details:
        print(f"  Graph built: {len(neighbors)} vertices, {edges} edges")
    step_timings['Step 5: Build neighbor graph'] = time.time() - step_start
    
    del struct_masks, struct_set_bits, index_map
    gc.collect()
    
    # ===== STEP 6: CHECK GRAPH CONNECTIVITY =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 6: CHECK STRUCTURE GRAPH CONNECTIVITY")
        print("-" * 50)
    
    graph_components = find_connected_components(neighbors)
    num_components = len(graph_components)
    component_sizes = [len(comp) for comp in graph_components]
    if show_details:
        print(f"  Connected components found: {num_components}")
        print(f"  Component sizes (first 15): {component_sizes[:15]}")
    step_timings['Step 6: Check graph connectivity'] = time.time() - step_start
    
    # ===== STEP 7: FIND LOCAL MINIMA (FOR ENTIRE GRAPH) =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 7: FIND LOCAL MINIMA")
        print("-" * 50)
    
    local_minima = find_local_minima(energies, neighbors)
    if show_details:
        print(f"Local minima found: {len(local_minima)}")
    step_timings['Step 7: Find local minima'] = time.time() - step_start
    
    # ===== STEP 8: DETERMINE MACROSTATES =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 8: DETERMINE MACROSTATES")
        print("-" * 50)
    
    basins = compute_gradient_basins(energies, neighbors, verbose=show_details)
    total_basins = len(basins)
    if show_details:
        print(f"Number of macrostates (before filtering): {total_basins}")
    step_timings['Step 8: Determine macrostates'] = time.time() - step_start
    
    # ===== STEP 9: COMPUTE PARTITION FUNCTIONS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 9: COMPUTE BASIN PARTITION FUNCTIONS")
        print("-" * 50)
    
    Z = {}
    for i, (_, indices) in enumerate(basins):
        Z[i] = sum(np.exp(-energies[idx] / RT) for idx in indices)
    
    if show_details:
        print(f"  Partition functions computed for {total_basins} basins")
    step_timings['Step 9: Compute partition functions'] = time.time() - step_start
    
    # ===== STEP 10: FILTER MACROSTATES (GLOBAL) =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 10: FILTER MACROSTATES")
        print("-" * 50)
    
    filtered_basins, old_to_new = filter_macrostates_spectral(
        basins, Z, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, verbose=show_details
    )
    num_filtered_basins = len(filtered_basins)
    if show_details:
        print(f"Number of macrostates (after filtering): {num_filtered_basins}")
    step_timings['Step 10: Filter macrostates'] = time.time() - step_start
    
    if num_filtered_basins < 3:
        print("Error: fewer than 3 macrostates remain after filtering")
        gc.collect()
        return None
    
    # ===== STEP 11: PROCESS BY CONNECTED COMPONENTS =====
    step_start = time.time()
    if show_details:
        print("\n" + "-" * 50)
        print("STEP 11: PROCESS BY CONNECTED COMPONENTS")
        print("-" * 50)
    
    if num_components == 1:
        if show_details:
            print(f"  Graph is connected. Processing as single component.")
        components_to_process = [(0, list(range(len(energies))))]
    else:
        if show_details:
            print(f"  Graph is disconnected ({num_components} components). Processing each separately.")
        components_to_process = []
        skipped_by_size = defaultdict(int)
        
        for comp_idx, comp_indices in enumerate(graph_components):
            comp_set = set(comp_indices)
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
        
        if show_details and skipped_by_size:
            print(f"    Components skipped (basins < 3):")
            for size in sorted(skipped_by_size.keys(), reverse=True):
                count = skipped_by_size[size]
                suffix = "" if count == 1 else "s"
                print(f"      size {size}: {count} component{suffix}")
    
    if not components_to_process:
        print("  Error: no connected component contains >= 3 basins")
        gc.collect()
        return None
    
    # ===== STEP 12: BUILD K_sym AND SPECTRAL ANALYSIS FOR EACH COMPONENT =====
    all_component_results = []
    global_basin_stats = []
    
    for comp_idx, comp_indices in components_to_process:
        if show_details:
            print(f"\n  --- Component {comp_idx} ({len(comp_indices)} structures) ---")
        
        comp_set = set(comp_indices)
        comp_energies = energies[list(comp_indices)]
        
        old_to_local = {old: local for local, old in enumerate(comp_indices)}
        
        comp_basins = []
        for basin_idx, (rep, basin_structs) in enumerate(filtered_basins):
            local_structs = [old_to_local[s] for s in basin_structs if s in comp_set]
            if len(local_structs) > 0:
                local_rep = local_structs[0]
                comp_basins.append((local_rep, local_structs))
        
        if len(comp_basins) < 3:
            if show_details:
                print(f"    Skipped: only {len(comp_basins)} basins (< 3)")
            continue
        
        if show_details:
            print(f"    Basins in component: {len(comp_basins)}")
        
        comp_neighbors = []
        for old_idx in comp_indices:
            local_idx = old_to_local[old_idx]
            local_nbs = set()
            for nb in neighbors[old_idx]:
                if nb in comp_set:
                    local_nbs.add(old_to_local[nb])
            comp_neighbors.append(local_nbs)
        
        if show_details:
            print(f"    Building K_sym ({len(comp_indices)} x {len(comp_indices)})...")
        K_sym_comp = build_transition_rate_matrix(
            comp_energies, comp_neighbors, temp_kelvin, FREQUENCY_PREFACTOR
        )
        if show_details:
            print(f"    Non-zero elements: {K_sym_comp.nnz}")
        
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
    
    step_timings['Steps 11-12: Component processing and spectral analysis'] = time.time() - step_start
    
    if not all_component_results:
        print("\n  Error: failed to process any connected component")
        gc.collect()
        return None
    
    total_triplets_all = sum(r['num_triplets'] for r in global_basin_stats)
    if total_triplets_all > 0:
        weighted_u_nt = sum(r['u_nt'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
        weighted_u_tr = sum(r['u_tr'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
        weighted_u_non = sum(r['u_non'] * r['num_triplets'] for r in global_basin_stats) / total_triplets_all
    else:
        weighted_u_nt = 0.0
        weighted_u_tr = 0.0
        weighted_u_non = 0.0
    
    # ===== COMPUTE INTER-COMPONENT TRIPLET FRACTION f_inter =====
    # f_inter = 1 - (sum of intra-family triplets) / (all basin triplets)
    K_total = num_filtered_basins
    if K_total >= 3:
        total_all_triplets = K_total * (K_total - 1) * (K_total - 2) // 6
        intra_triplets = total_triplets_all  # sum of C(K_c, 3) over processed components
        f_inter = 1.0 - (intra_triplets / total_all_triplets) if total_all_triplets > 0 else 0.0
    else:
        f_inter = 0.0
    
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
        print(f"  Inter-component triplet fraction (f_inter): {f_inter:.4f}")
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
    
    first_res = all_component_results[0]
    
    return {
        'sequence': seq[:50],
        'description': seq_description,
        'length': seq_len,
        'weighted_u_nt': weighted_u_nt,
        'weighted_u_tr': weighted_u_tr,
        'weighted_u_non': weighted_u_non,
        'f_inter': f_inter,  # inter-component triplet fraction
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
# SEQUENCE PROCESSING FUNCTION (WITH NUM_STAT SUPPORT)
# ============================================================================

def process_sequence(seq, seq_description, seq_index, total_sequences):
    """
    Performs full ultrametricity analysis for one RNA sequence.
    """
    all_runs = []
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    temp_kelvin = TEMPERATURE_CELSIUS + 273.15
    RT = R_KCAL * temp_kelvin
    
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
    print(f"Max macrostates: {MAX_MACROSTATES_ANALYSIS}")
    print(f"Requested eigenmodes: {NUM_EIGENMODES}")
    print(f"Spectral gap threshold: {SPECTRAL_GAP_THRESHOLD:.1e}")
    print(f"Frequency prefactor nu0: {FREQUENCY_PREFACTOR}")
    print(f"ARPACK max iterations: {EIGS_MAXITER}")
    print(f"ARPACK shift sigma: {EIGS_SIGMA}")
    print(f"Min hairpin length: {MIN_HAIRPIN_LEN} (j - i >= {MIN_HAIRPIN_LEN + 1})")
    print(f"Tolerance epsilon: {ULTRAMETRIC_EPSILON}, delta: {ULTRAMETRIC_DELTA}")
    print(f"Seed: {RANDOM_SEED} (base)")
    if NUM_STAT > 1:
        print(f"NUM_STAT: {NUM_STAT} (statistical trials)")
    print(f"Number of processes (CPU): {n_workers}")
    print(f"Method: spectral Mahalanobis distance with auto noise filtering")
    print(f"\nR*T = {RT:.6f} kcal/mol")
    
    for run_idx in range(NUM_STAT):
        current_seed = RANDOM_SEED + run_idx
        
        if not VERBOSE and NUM_STAT > 1:
            print(f"\nRUN {run_idx + 1}/{NUM_STAT}")
        
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
        res = all_runs[0]
        print("\n" + "=" * 70)
        print(f"RESULT FOR SEQUENCE {seq_index}")
        print("=" * 70)
        print(f"  Weighted u_nt:              {res['weighted_u_nt']:.2f} %")
        print(f"  Weighted u_tr:              {res['weighted_u_tr']:.2f} %")
        print(f"  Weighted u_non:             {res['weighted_u_non']:.2f} %")
        print(f"  Inter-component fraction:   {res['f_inter']:.4f}")
        print(f"  Number of structures:       {res['n_structures']}")
        print(f"  Number of basins:           {res['n_basins']}")
        print(f"  Connected components:       {res['n_components']}")
        print(f"  Elapsed time:               {res['elapsed']:.1f} sec")
        return res
    
    print("\n" + "=" * 70)
    print(f"STATISTICS OVER {len(all_runs)} RUNS FOR SEQUENCE {seq_index}")
    print("=" * 70)
    
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
    
    mean_f_inter = np.mean(f_inter_vals)
    std_f_inter = np.std(f_inter_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_struct = np.mean(n_structures_vals)
    std_struct = np.std(n_structures_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_basins = np.mean(n_basins_vals)
    std_basins = np.std(n_basins_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_comp = np.mean(n_components_vals)
    std_comp = np.std(n_components_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_elapsed = np.mean(elapsed_vals)
    std_elapsed = np.std(elapsed_vals, ddof=1) if len(all_runs) > 1 else 0.0
    
    mean_struct_rounded = int(round(mean_struct))
    std_struct_rounded = int(round(std_struct))
    mean_basins_rounded = int(round(mean_basins))
    std_basins_rounded = int(round(std_basins))
    mean_comp_rounded = int(round(mean_comp))
    std_comp_rounded = int(round(std_comp))
    
    print(f"  Weighted u_nt:              {mean_u_nt:.2f} +/- {std_u_nt:.2f} %")
    print(f"  Weighted u_tr:              {mean_u_tr:.2f} +/- {std_u_tr:.2f} %")
    print(f"  Weighted u_non:             {mean_u_non:.2f} +/- {std_u_non:.2f} %")
    print(f"  Inter-component fraction:   {mean_f_inter:.4f} +/- {std_f_inter:.4f}")
    print(f"  Number of structures:       {mean_struct_rounded} +/- {std_struct_rounded}")
    print(f"  Number of basins:           {mean_basins_rounded} +/- {std_basins_rounded}")
    print(f"  Connected components:       {mean_comp_rounded} +/- {std_comp_rounded}")
    print(f"  Elapsed time:               {mean_elapsed:.1f} +/- {std_elapsed:.1f} sec")
    
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
        'f_inter': mean_f_inter,
        'f_inter_std': std_f_inter,
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
        'all_runs': all_runs
    }


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    total_start_time = time.time()
    
    print("=" * 70)
    print("NONTRIVIAL ULTRAMETRICITY DEGREE CALCULATION")
    print("FOR RNA SECONDARY STRUCTURE MACROSTATES")
    print("METHOD: spectral Mahalanobis distance")
    print("(physically rigorous approach via transition rate matrix)")
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
    
    if len(sequences) > 1 and all_seq_results:
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        
        if NUM_STAT > 1:
            header = (f"{'No.':<4} {'Description':<30} {'Len':<8} "
                      f"{'Structures':<16} {'Comp':<14} {'Basins':<16} "
                      f"{'f_inter':<14} "
                      f"{'u_nt (%)':<18} {'u_tr (%)':<18} {'u_non (%)':<18} {'Time (s)':<16}")
        else:
            header = (f"{'No.':<4} {'Description':<35} {'Len':<8} {'Struct':<10} "
                      f"{'Comp':<6} {'Basins':<12} "
                      f"{'f_inter':<10} "
                      f"{'u_nt (%)':<12} {'u_tr (%)':<12} {'u_non (%)':<12} {'Time (s)':<10}")
        print(header)
        print("-" * len(header))
        
        for i, res in enumerate(all_seq_results):
            desc = res['description'][:28] if NUM_STAT > 1 else res['description'][:33]
            
            if NUM_STAT > 1:
                struct_str = f"{int(res['n_structures'])}+/-{int(res['n_structures_std'])}"
                comp_str = f"{int(res['n_components'])}+/-{int(res['n_components_std'])}"
                basins_str = f"{int(res['n_basins'])}+/-{int(res['n_basins_std'])}"
                f_inter_str = f"{res['f_inter']:.4f}+/-{res['f_inter_std']:.4f}"
                u_nt_str = f"{res['weighted_u_nt']:.2f}+/-{res['weighted_u_nt_std']:.2f}"
                u_tr_str = f"{res['weighted_u_tr']:.2f}+/-{res['weighted_u_tr_std']:.2f}"
                u_non_str = f"{res['weighted_u_non']:.2f}+/-{res['weighted_u_non_std']:.2f}"
                time_str = f"{res['elapsed']:.1f}+/-{res['elapsed_std']:.1f}"
                print(f"{i+1:<4} {desc:<30} {res['length']:<8} "
                      f"{struct_str:<16} {comp_str:<14} {basins_str:<16} "
                      f"{f_inter_str:<14} "
                      f"{u_nt_str:<18} {u_tr_str:<18} {u_non_str:<18} {time_str:<16}")
            else:
                u_nt_str = f"{res['weighted_u_nt']:.2f}"
                u_tr_str = f"{res['weighted_u_tr']:.2f}"
                u_non_str = f"{res['weighted_u_non']:.2f}"
                f_inter_str = f"{res['f_inter']:.4f}"
                print(f"{i+1:<4} {desc:<35} {res['length']:<8} {res['n_structures']:<10} "
                      f"{res['n_components']:<6} {res['n_basins']:<12} "
                      f"{f_inter_str:<10} "
                      f"{u_nt_str:<12} {u_tr_str:<12} {u_non_str:<12} {res['elapsed']:<10.1f}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {total_elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
