
# ============================================================================
# Version 9
# CRITICAL: BLAS/LAPACK/OpenMP/MKL Thread Limitation
# On Windows os.environ is NOT inherited by child processes on spawn!
# Therefore, we set limits BEFORE importing numpy/scipy AND
# additionally limit via mkl/threadpoolctl inside workers.
# ============================================================================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENAI_NUM_THREADS'] = '1'

# Attempt to limit threads via threadpoolctl (works on Windows!)
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api='blas')
    threadpool_limits(limits=1, user_api='openmp')
except ImportError:
    pass  # If threadpoolctl is not installed, rely on os.environ

"""
Calculation of the degree of nontrivial ultrametricity for RNA macrostates.
PHYSICALLY RIGOROUS APPROACH: distance between basins via spectral
decomposition of the transition rate matrix (Mahalanobis distance
in the space of eigenvectors of the symmetrized matrix K).

METHOD:
1. The transition rate matrix K between all structures is built
   (N x N, where N ~ 2000) based on the Kramers formula.
2. K is symmetrized taking into account detailed balance.
   [FIXED] Diagonal elements of S are computed via the original
   asymmetric matrix K to preserve detailed balance.
3. The m smallest eigenvalues in magnitude and corresponding
   eigenvectors are computed (Lanczos method for sparse matrices).
4. Automatic filtering of noise modes is performed by searching
   for a spectral gap: if the ratio |lambda_k| / |lambda_{k-1}| exceeds
   a threshold (default 10^6), modes with indices < k are discarded
   as numerical noise.
5. Each attraction basin is represented by a characteristic
   vector chi_A in the space of structures.
   [FIXED] Vectors are built using the Boltzmann distribution:
   u_a(p) = w^{1/2}(p) / W_a, where w(p) = exp(-G/RT),
   and W_a is the partition function of the basin (taking into account
   the shift by G_min to prevent overflow, as stated in Stage 9 of the paper).
6. The distance between basins A and B is defined as the weighted
   Euclidean distance between the projections of chi_A and chi_B onto the
   eigenvectors (Mahalanobis distance).
7. The resulting distance matrix is a metric and is tested for ultrametricity.

HANDLING DISCONNECTED GRAPHS:
Before building the K_sym matrix, the connectivity of the structure graph is checked.
If the graph contains multiple connected components, each component is processed
separately: its own K_sym matrix is built, spectral decomposition and ultrametricity
testing are performed.
Components with less than 3 basins are skipped.
Components containing less than ALPHA_COMPONENT_THRESHOLD * N structures are
classified as noise and excluded from the f_inter calculation and final spectral analysis.

IMPORTANT: This logic of processing ALL significant components with subsequent
weighted averaging is applied UNIFORMLY both in the main stage and in all
null hypothesis testing modes (nt_shuffle, energy_shuffle, topo_shuffle).
This guarantees the statistical reliability of the results.

STATISTICAL MODE (NUM_STAT > 1):
When NUM_STAT > 1, NUM_STAT independent runs with different random structure
samples are performed for each sequence (seed varies: RANDOM_SEED, RANDOM_SEED+1,
..., RANDOM_SEED+NUM_STAT-1).
Runs are executed IN PARALLEL via multiprocessing.Pool for maximum utilization
of computational resources.
Results are averaged, and the final table displays mean values and standard
deviations (mean +/- std). Integer quantities (number of structures, basins,
connected components) are rounded to integers.

OUTPUT MODES:
VERBOSE = True  - full log (steps, components, spectral analysis,
                  detailed time breakdown by stages in the main stage
                  and in null models).
VERBOSE = False - brief log: sequence header and parameters are printed once,
                  then only LAUNCH/COMPLETED, then the statistics block.

NULL HYPOTHESIS TESTING (NULL_MODEL_TYPE):
Testing is carried out by comparing the real system with null models that differ
in the degree of "randomness". All null models are executed IN PARALLEL and process
ALL significant connected components.
Statistical significance is estimated via a TWO-SIDED p-value, since biological
function may require both the presence of pronounced hierarchy (high ultrametricity)
and its absence or specific frustration (low ultrametricity). The two-sided test
checks the significance of the deviation of the real value from the mean of the
null ensemble in both directions.

'none'            : The program runs in normal mode without tests.
'full_analysis'   : (RECOMMENDED) FULL MECHANISM ANALYSIS.
                    Automatically performs TWO independent tests:
                    1. Energy Shuffle: Graph preservation + energy shuffling.
                       Shows the contribution of pure graph topology.
                    2. Topo Shuffle: Configuration model (edge rewiring preserving
                       vertex degrees) + energy shuffling. Shows the baseline chaos level.
                    Outputs TWO separate summary tables for each test.
'topo_shuffle'    : CONFIGURATION MODEL.
                    Graph edge rewiring via double_edge_swap while preserving
                    the vertex degree sequence + energy shuffling. Destroys topological
                    correlations while preserving mobility distribution.
                    Basins are found anew for ALL significant components.
'energy_shuffle'  : (WEAK RANDOMNESS / TOPOLOGICAL ORDER)
                    The neighborhood graph is fully preserved (including all
                    topological correlations), but vertex energies are randomly shuffled.
                    Basins are found anew for these random energies in ALL significant
                    components. Allows isolating the contribution of PURE GRAPH TOPOLOGY
                    to ultrametricity.
'nt_shuffle'      : NUCLEOTIDE SHUFFLING (BIOLOGICAL CONTROL).
                    Random permutations of the original RNA sequence are generated
                    preserving its nucleotide composition. For each permutation, all
                    stages are completely re-executed: structure generation, graph building,
                    search for ALL significant components and basins, spectral analysis.
                    This is the strictest test checking whether ultrametricity is determined
                    by the specific nucleotide order. REQUIRES A LOT OF TIME.
'random_basins'   : (GEOMETRIC CONTROL)
                    The real spectrum of the K_sym matrix is preserved, but structures
                    are randomly partitioned into basins of the same sizes.
                    Checks whether ultrametricity is an artifact of the high-dimensional
                    eigenvector space geometry. Does not affect the graph or energies.

NULL MODEL OPTIMIZATION:
For the energy_shuffle mode, data on connected components (lists of neighbors
in local numbering) are prepared ONCE before launching the process pool.
This avoids multiple serializations of the full neighborhood graph and
rebuilding neighbor lists inside each worker, which significantly speeds up
computations compared to nt_shuffle and topo_shuffle, where the graph changes
or is generated anew in each run.

EXPECTATION OVER RNA ENSEMBLE (EXPECTATION_BY_RNA = True):
If EXPECTATION_BY_RNA = True, a line "AVERAGE OVER ALL RNAs" is added to the
end of EACH summary table (main and all null model tables), containing the
mean values and STD of all corresponding metrics over the entire set of sequences.
This allows estimating typical values and spread over the ensemble for both
real data and each null model.

ADVANTAGES:
- Takes into account all possible transition paths (via spectral decomposition).
- Context-independent (distance between A and B is determined only by them,
  not by the presence of other basins).
- Symmetric and guaranteed to be a metric.
- Automatically filters out numerical noise via spectral gap search.
- Correctly handles disconnected structure graphs.
- Computational complexity O(m*N*E + K^2*m), allowing processing of
  N ~ 2000 structures and K ~ 100 basins in seconds.
- Full parallelization of the main stage and all null models.
- Unified methodology for component processing in all modes.
- Two-sided statistical significance testing.
- IPC optimization for energy_shuffle via preliminary data preparation.
- Detailed execution time diagnostics by stages when VERBOSE=True
  for the main stage and all null models.

STRUCTURE GENERATION MODE:
Stochastic sampling (pbacktrack) from the Gibbs distribution.

Dependencies: pip install viennarna numpy scipy biopython threadpoolctl
"""

import RNA
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
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
"""Global variable: dictionary {bitmask: index} for O(1) search in workers."""
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
True  - scan the current folder for *.fasta files,
        load all sequences, sort by length.
False - use the sequence from RNA_SEQUENCE.
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
Low (< 20 C): deep basins, rare transitions.
High (> 60 C): smoothed landscape, fast transitions.
Recommended value: 37.0 (physiological temperature).
Allowed range: 0.0 - 100.0.
"""

ENERGY_WINDOW = 50.0
"""
Energy window (kcal/mol) relative to the minimum free energy (MFE).
During stochastic sampling, structures with energy > MFE + ENERGY_WINDOW
are discarded. If set to "inf" - no energy window.
Small window (1-5 kcal/mol): only the most stable structures, might not be enough for analysis.
Large window (> 15 kcal/mol): many structures, graph is sparse, computation time grows.
Recommended value: 10.0.
Allowed range: positive number or "inf".
"""

# --- Structure generation parameters ---
MAX_STRUCTURES = 100000
"""
Maximum number of generated secondary structures (microstates).
Generation stops when the number of unique structures in the specified
energy window reaches this value.
Few (100-500): fast, but statistically poor analysis.
Many (> 10000): full landscape picture, but slow.
Recommended value: 2000-5000.
Allowed range: 100 - 20000.
"""

MIN_HAIRPIN_LEN = 3
"""
Minimum number of unpaired nucleotides in a hairpin loop.
Defines the condition: j - i - 1 >= MIN_HAIRPIN_LEN.
Standard value: 3 (steric constraint).
Value 0 disables the constraint (unphysical).
Recommended value: 3.
Allowed range: 0 - 10.
"""

RANDOM_SEED = 43
"""
Initial value for the random number generator.
Ensures reproducibility of results.
When NUM_STAT > 1, seed varies: RANDOM_SEED, RANDOM_SEED+1, ...
Recommended value: 42 (or any integer).
Allowed range: any integer.
"""

# --- Attraction basin parameters ---
MAX_MACROSTATES_ANALYSIS = 500
"""
Maximum number of attraction basins participating in the final analysis.
If more remain after filtering, basins with the highest partition functions Z are kept.
Few (10-30): fast, but might not be enough for triplet statistics.
Many (> 200): more triplets for analysis, but slower (K^3 for spectrum).
Recommended value: 100.
Allowed range: 3 - 500.
"""

MIN_MACROSTATE_SIZE = 5
"""
Minimum size of an attraction basin (number of included structures).
Basins smaller than this are considered statistically insignificant.
Value 1: all basins are included, including isolated structures.
Value 5-10: small artifactual basins are filtered out.
Recommended value: 5.
Allowed range: 1 - 100.
"""

# --- Connected component filtering parameter ---
ALPHA_COMPONENT_THRESHOLD = 0.001
"""
Relative threshold for classifying connected components of the structure graph
as significant or noise. A component is considered significant if it contains
at least max(3, ALPHA_COMPONENT_THRESHOLD * N) structures, where N is the total
number of unique structures in the sample. Noise components are excluded from
the f_inter calculation and final spectral analysis.
Small value (0.001): conservative, artifactual components may remain.
Large value (0.05): aggressive, real small families might be lost.
Recommended value: 0.01.
Allowed range: 0.001 - 0.1.
"""

# --- Spectral analysis parameters ---
NUM_EIGENMODES = 50
"""
Number of eigenmodes (eigenvalues and eigenvectors) requested for spectral decomposition.
After automatic noise mode filtering, the actual number of used modes may be smaller.
Few (5-10): fast, but information about fine landscape structure is lost.
Many (> 100): more accurate, but slower (grows linearly).
Constraint: must be strictly less than the number of structures.
Recommended value: 50.
Allowed range: 5 - 200 (but not more than N-2, where N is the number of structures).
"""

SPECTRAL_GAP_THRESHOLD = 1e6
"""
Threshold for detecting a spectral gap between noise and physical modes.
If the ratio |lambda_k| / |lambda_{k-1}| > SPECTRAL_GAP_THRESHOLD, modes with indices
< k are considered numerical noise and discarded.
Large threshold (10^8): conservative, weak physical modes might be lost.
Small threshold (10^2): aggressive, noise modes might be kept.
Recommended value: 1e6.
Allowed range: 1e2 - 1e12.
"""

FREQUENCY_PREFACTOR = 1.0
"""
Frequency prefactor nu_0 in the Kramers formula (in arbitrary units).
Affects the absolute scale of matrix K, but does not affect eigenvectors
and relative distances between basins (changing nu_0 multiplies all lambda_k
by a constant, which cancels out in the Mahalanobis distance).
Recommended value: 1.0 (leave unchanged).
Allowed range: any positive number.
"""

EIGS_MAXITER = 50000
"""
Maximum number of iterations for the Lanczos algorithm (ARPACK) when computing
eigenvalues of the K_sym matrix. Increasing this parameter improves convergence
for matrices with a dense spectrum near zero, but increases computation time.
Recommended value: 50000.
Allowed range: 1000 - 200000.
"""

EIGS_SIGMA = 1e-4
"""
Shift sigma for the Lanczos algorithm (shift-invert) when searching for eigenvalues
near zero.
Matrix K_sym is negative semi-definite (all lambda <= 0).
Using shift-invert (which='LM', sigma > 0) allows instantly finding modes close to zero.
Too small value (1e-10): matrix (K - sigma*I) is nearly singular, LU factorization
(SuperLU) "hesitates" for tens of seconds due to numerical instability and pivoting.
Optimal value (1e-4 ... 1e-5): shift enhances diagonal dominance, factorization
takes fractions of a second even for random graphs.
Recommended value: 1e-4.
Allowed range: 1e-6 - 1e-3.
"""

# --- Ultrametricity testing parameters ---
ULTRAMETRIC_EPSILON = 0.05
"""
Relative precision epsilon for testing approximate ultrametricity.
The two largest sides of a triangle are considered equal if
(d_max - d_mid) / d_mid <= epsilon.
Must be strictly less than ULTRAMETRIC_DELTA.
When epsilon = 0: exact equality is required (almost unattainable).
When epsilon > 0.1: many false positive classifications.
Recommended value: 0.05.
Allowed range: 0.0 - 0.20.
"""

ULTRAMETRIC_DELTA = 0.1
"""
Minimum relative difference delta between the smaller and middle sides
of a triangle to be classified as nontrivially ultrametric:
(d_mid - d_min) / d_mid > delta.
Must be strictly greater than ULTRAMETRIC_EPSILON.
When delta is small: equilateral triangles are erroneously classified
as nontrivially ultrametric.
When delta is large: almost no nontrivially ultrametric triplets remain.
Recommended value: 0.1.
Allowed range: 0.01 - 0.50.
"""

# --- Numerical precision parameters ---
EPS_COMPARISON = 1e-9
"""
Threshold for comparing real numbers (energies, distances).
Used to check strict inequalities in plateau conditions, local minima,
and triangle classification.
Too small (< 1e-12): risk of false distinction due to rounding noise.
Too large (> 1e-6): risk of merging distinct states.
Recommended value: 1e-9.
Allowed range: 1e-12 - 1e-6.
"""

# --- Computational resource parameters ---
NUM_WORKERS = 6
"""
Number of parallel processes for generating neighboring structures and
executing NUM_STAT runs / null models.
IMPORTANT: It is recommended to set the value less than the number of physical cores
(e.g., 12 for a 24-thread CPU) to leave resources for system processes and avoid
overload due to internal BLAS/LAPACK multithreading.
None: automatically use all available CPU cores.
1: single-threaded mode (for debugging).
N: use exactly N processes.
Recommended value: 12 (or cpu_count() // 2).
Allowed range: 1 - cpu_count().
"""

VERBOSE = True
"""
Verbose output mode.
True: output of all intermediate results (basin sizes, transition statistics,
      triangle distribution, detailed time breakdown by stages in the main stage
      and in null models).
False: only final results (brief log).
Recommended value: True (for research purposes).
"""

# --- Statistical analysis parameter ---
NUM_STAT = 5
"""
Number of statistical trials (independent runs) for each RNA sequence.
Runs are executed IN PARALLEL.
NUM_STAT = 1: single run, result without deviation.
NUM_STAT > 1: NUM_STAT runs are executed with different seeds
(RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1).
Results are averaged, output as mean +/- STD.
Integer quantities (number of structures, basins, components) are rounded to integers.
Recommended value: 1.
Allowed range: 1 - 100.
"""

# --- Null hypothesis testing parameters ---
NULL_MODEL_TYPE = 'nt_shuffle'
"""
Type of null model for testing the hypothesis about the origin of ultrametricity.
All models are executed IN PARALLEL and process ALL significant components.
Statistical significance is estimated via a TWO-SIDED p-value.

'none'          : No null hypothesis testing.
'full_analysis' : (RECOMMENDED) Full mechanism analysis.
                  Performs TWO tests: energy_shuffle and topo_shuffle.
                  Outputs two separate summary tables.
'topo_shuffle'  : Configuration model. Graph edge rewiring
                  (double_edge_swap preserving vertex degrees) + energy shuffling.
                  Destroys topological correlations, preserving mobility distribution.
'energy_shuffle': Neighborhood graph preservation (including all topological
                  correlations), energy shuffling. Contribution of pure graph
                  topology to ultrametricity.
'nt_shuffle'    : Nucleotide shuffling. Full recalculation of structures and graph
                  for random sequence permutations. Strictest biological control.
'random_basins' : Geometric control. Real spectrum, random basins of the same sizes.
                  Checks for space artifacts.

Recommended value: 'full_analysis'.
"""

NUM_NULL_SAMPLES = 100
"""
Number of runs for each null hypothesis test.
Executed IN PARALLEL via multiprocessing.Pool.
For 'random_basins' you can set 100-500 (very fast).
For 'energy_shuffle', 'topo_shuffle' 20-30 is recommended.
For 'nt_shuffle' 5-10 is recommended (very slow, full recalculation).
In 'full_analysis' mode, this number applies to both tests.
"""

NUM_EDGE_SWAPS_MULTIPLIER = 10
"""
Multiplier for the number of edge swaps during graph rewiring
(topo_shuffle and full_analysis modes).
Number of swaps = NUM_EDGE_SWAPS_MULTIPLIER * |E|.
Small (1-3): fast mixing, but topological correlations might not fully break down.
Medium (5-10): good balance of speed and mixing quality.
Large (>20): thorough destruction of correlations, but slower.
Recommended value: 10.
Allowed range: 1 - 100.
"""

# --- RNA ensemble averaging parameters ---
EXPECTATION_BY_RNA = False
"""
Mode for outputting summary statistics over all sequences.
False: program runs without adding a summary line.
True:  a line "AVERAGE OVER ALL RNAs" is added to the end of EACH summary table
       (main and all null model tables), containing the mean values and STD
       of all corresponding metrics over the entire set of sequences.
Recommended value: False (enable for generalized evaluation).
"""

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
R_KCAL = 0.001987204259  # Gas constant in kcal/(mol*K) (R = N_A * k_B)

# ============================================================================
# HELPER FUNCTION FOR LIMITING THREADS INSIDE WORKERS
# ============================================================================
def _limit_threads_in_worker():
    """
    CRITICAL FIX FOR WINDOWS: limiting BLAS/OpenMP threads INSIDE each
    multiprocessing worker.
    On Windows, os.environ from the parent process is NOT inherited on spawn,
    so limits must be set directly in each worker.
    """
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api='blas')
        threadpool_limits(limits=1, user_api='openmp')
    except ImportError:
        pass

# ============================================================================
# OPTIMIZATION: BITMASKS AND PRECOMPUTATION OF ALLOWED PAIRS + CONFLICTS
# ============================================================================
def precompute_allowed_pairs_and_conflicts(seq_len, sequence, min_hairpin_len, comp_map):
    """
    Precomputes the list of all allowed pairs and the conflict matrix between them.
    Conflicts are encoded as bitmasks for O(1) checking.
    Returns:
    allowed (list): list of pairs (i, j)
    pair_to_idx (dict): mapping of a pair to its index
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
    
    # Precompute powers of two to accelerate bitwise operations
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
            
            # Check for shared nucleotides or intersection (pseudoknot)
            if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
                mask |= bit[idx2]
            elif (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                mask |= bit[idx2]
        conflict_masks[idx1] = mask
        
    return allowed, pair_to_idx, conflict_masks, bit, P

# ============================================================================
# FUNCTIONS FOR WORKING WITH STRUCTURES
# ============================================================================
def dotbracket_to_pairs(structure):
    """
    Converts a dot-bracket structure into a set of base pairs.
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
    Converts a dot-bracket structure into a bitmask of allowed pair indices.
    Returns the mask and a tuple of set bits (for fast iteration).
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
    Removes duplicate structures (with identical sets of pairs).
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
    Initializer for processes in Pool.
    Sets module global variables for O(1) access.
    ON WINDOWS: Also forcibly limits BLAS/OpenMP threads, since os.environ
    from the parent process is NOT inherited on spawn.
    """
    # CRITICAL FIX FOR WINDOWS:
    # Limit BLAS/OpenMP threads INSIDE each worker
    _limit_threads_in_worker()
    
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
    
    # Operation 1: Remove an existing pair
    for idx_out in set_bits:
        new_mask = mask & ~_BIT[idx_out]
        nb_idx = _INDEX_MAP.get(new_mask)
        if nb_idx is not None:
            neighbors.append(nb_idx)
            
    # Operation 2: Add a new pair
    for idx_in in range(_P):
        if not (mask & _BIT[idx_in]):
            # O(1) conflict check via bitwise AND
            if (mask & _CONFLICT_MASKS[idx_in]) == 0:
                new_mask = mask | _BIT[idx_in]
                nb_idx = _INDEX_MAP.get(new_mask)
                if nb_idx is not None:
                    neighbors.append(nb_idx)
                    
    # Operation 3: Pair shift (removal + addition)
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
    Local version of graph building, not using global variables.
    Necessary for correct operation of nt_shuffle in multiprocessing,
    where each worker has its own sequence and graph.
    """
    n_structures = len(struct_masks)
    neighbors_list = [set() for _ in range(n_structures)]
    
    for idx in range(n_structures):
        mask = struct_masks[idx]
        set_bits = struct_set_bits[idx]
        
        # Operation 1: Remove an existing pair
        for idx_out in set_bits:
            new_mask = mask & ~bit[idx_out]
            nb_idx = index_map.get(new_mask)
            if nb_idx is not None:
                neighbors_list[idx].add(nb_idx)
                
        # Operation 2: Add a new pair
        for idx_in in range(P):
            if not (mask & bit[idx_in]):
                if (mask & conflict_masks[idx_in]) == 0:
                    new_mask = mask | bit[idx_in]
                    nb_idx = index_map.get(new_mask)
                    if nb_idx is not None:
                        neighbors_list[idx].add(nb_idx)
                        
        # Operation 3: Pair shift (removal + addition)
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
# GRAPH BUILDING AND ANALYSIS FUNCTIONS
# ============================================================================
def generate_structures_stochastic(seq, temp_celsius, max_structures, energy_window, verbose=True):
    """
    Generation of a set of secondary structures by stochastic sampling
    from the Boltzmann ensemble (pbacktrack) with energy limitation.
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
    Builds the neighborhood graph based on neighbor generation via bitmasks.
    Uses optimized IPC (returns only indices).
    """
    n_workers = num_workers if num_workers else cpu_count()
    n_structures = len(struct_masks)
    neighbors_list = [set() for _ in range(n_structures)]
    
    if n_workers > 1:
        if verbose:
            print(f"  Using {n_workers} processes for neighbor generation (Bitmask O(1) IPC)")
            
        indexed_args = [(idx, struct_masks[idx], struct_set_bits[idx]) for idx in range(n_structures)]
        
        # FIX: maxtasksperchild=1 to prevent memory leaks in long-lived multiprocessing processes.
        # After each task, the process is restarted, guaranteeing the release of memory allocated
        # by C-extensions (SuperLU, ARPACK).
        with Pool(n_workers, initializer=_pool_initializer_bitmask,
                  initargs=(index_map, conflict_masks, bit, P),
                  maxtasksperchild=1) as pool:
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
    Determines attraction basins (gradient basins) with correct plateau handling.
    """
    n = len(energies)
    
    # Preliminarily find the strict descent direction for each node
    min_neighbor = [-1] * n
    for i in range(n):
        nbs = neighbors_list[i]
        if not nbs:
            min_neighbor[i] = -1
        else:
            best_nb = min(nbs, key=lambda x: (energies[x], x))
            if best_nb == i or energies[best_nb] >= energies[i] - EPS_COMPARISON:
                min_neighbor[i] = -1 # No strict descent
            else:
                min_neighbor[i] = best_nb
                
    candidate_set = {i for i in range(n) if min_neighbor[i] == -1}
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
    
    # FIX: Iterative traversal (Path Compression) instead of recursion
    for i in range(n):
        if basin_of[i] != -1 or i in attraction_id:
            continue
        path = []
        curr = i
        while basin_of[curr] == -1 and curr not in attraction_id:
            path.append(curr)
            nxt = min_neighbor[curr]
            if nxt == -1: break
            curr = nxt
            
        target_basin = attraction_id[curr] if curr in attraction_id else basin_of[curr]
        for node in path:
            basin_of[node] = target_basin
            
    for i in range(n):
        if basin_of[i] == -1 and i in attraction_id:
            basin_of[i] = attraction_id[i]
            
    basins_dict = defaultdict(list)
    for idx, b in enumerate(basin_of):
        if b != -1: basins_dict[b].append(idx)
        
    basins = [(attraction_points[b][0], indices) for b, indices in basins_dict.items() if b < len(attraction_points)]
    basins.sort(key=lambda x: energies[x[0]])
    
    if verbose:
        print(f"  Number of macrostates (basins): {len(basins)}")
        
    return basins

def build_transition_rate_matrix(energies, neighbors_list, temp_kelvin, nu0):
    """
    Builds the symmetrized transition rate matrix K_sym.
    FIXED: Diagonal elements are computed via the original asymmetric matrix K,
    ensuring exact correspondence to the Markov process and preservation of detailed balance.
    """
    N = len(energies)
    RT = R_KCAL * temp_kelvin
    
    rows_S, cols_S, data_S = [], [], []
    rows_K, cols_K, data_K = [], [], []
    
    for p in range(N):
        G_p = energies[p]
        for q in neighbors_list[p]:
            if q > p:
                G_q = energies[q]
                
                # 1. Symmetrized off-diagonal elements S (formula 6 from the paper)
                S_pq = nu0 * np.exp(-abs(G_p - G_q) / (2.0 * RT))
                rows_S.extend([p, q])
                cols_S.extend([q, p])
                data_S.extend([S_pq, S_pq])
                
                # 2. Elements of the original asymmetric matrix K (formula 4 from the paper)
                max_G = max(G_p, G_q)
                K_pq = nu0 * np.exp(-(max_G - G_p) / RT)
                K_qp = nu0 * np.exp(-(max_G - G_q) / RT)
                rows_K.extend([p, q])
                cols_K.extend([q, p])
                data_K.extend([K_pq, K_qp])
                
    # Form sparse matrix K for correct sum(axis=1)
    K_asym = csr_matrix((data_K, (rows_K, cols_K)), shape=(N, N))
    diag_vals = -np.array(K_asym.sum(axis=1)).flatten()
    
    # Form sparse matrix S and set diagonal
    K_sym = csr_matrix((data_S, (rows_S, cols_S)), shape=(N, N))
    K_sym.setdiag(diag_vals)
    K_sym.eliminate_zeros()
    
    return K_sym

def filter_eigenvalues_by_gap(eigenvalues, eigenvectors, gap_threshold):
    """
    Automatically finds the spectral gap and filters out noise modes.
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
            print(f"  Kept macrostates with highest Z: {len(valid)} (out of {len(basins)})")
    else:
        valid.sort(key=lambda i: Z[i], reverse=True)
        if verbose:
            print(f"  Kept macrostates: {len(valid)}")
            
    return [basins[i] for i in valid], {old: new for new, old in enumerate(valid)}

def compute_spectral_distance(K_sym, basins, energies, num_modes_requested, temp_kelvin, gap_threshold, eigs_maxiter, eigs_sigma, verbose=True):
    """
    Computes the Mahalanobis distance between basins.
    FIXED: Macrostate vectors are built using the Boltzmann distribution:
    u_a(p) = w^{1/2}(p) / W_a (according to the paper).
    Also returns the number of found physical modes for logging.
    """
    N = K_sym.shape[0]
    K_basins = len(basins)
    num_modes_requested = min(num_modes_requested, N - 1)
    ncv = min(2 * num_modes_requested + 10, N)
    
    eigenvalues, eigenvectors = None, None
    last_error = None
    
    # FIX: Use shift-invert with sigma=eigs_sigma (1e-4) and which='LM'.
    try:
        eigenvalues, eigenvectors = eigsh(K_sym, k=num_modes_requested, which='LM', sigma=eigs_sigma,
                                          return_eigenvectors=True, maxiter=eigs_maxiter, ncv=ncv, tol=1e-5)
    except Exception as e:
        last_error = e
        
    # Fallback in case shift-invert fails
    if eigenvalues is None:
        try:
            eigenvalues, eigenvectors = eigsh(K_sym, k=num_modes_requested, which='SM',
                                              return_eigenvectors=True, maxiter=eigs_maxiter, ncv=ncv, tol=1e-5)
        except Exception as e2:
            last_error = e2
            
    if eigenvalues is None:
        raise RuntimeError(f"Failed to compute eigenvalues. Last error: {last_error}")
        
    eigenvalues_filtered, eigenvectors_filtered, num_noise = filter_eigenvalues_by_gap(eigenvalues, eigenvectors, gap_threshold)
    num_phys = len(eigenvalues_filtered)
    
    if num_phys == 0:
        raise RuntimeError("All eigenmodes filtered as noise.")
        
    RT = R_KCAL * temp_kelvin
    
    # FIXED: Compute vector u_a according to the paper: u_a(p) = w^{1/2}(p) / W_a
    # where w(p) = exp(-G(p)/RT), W_a = sum_{q in M_a} w(q).
    # To prevent overflow, subtract the minimum energy (Stage 9 of the paper).
    min_e = np.min(energies)
    w = np.exp(-(energies - min_e) / RT)
    w_half = np.exp(-(energies - min_e) / (2.0 * RT))
    
    chi = np.zeros((K_basins, N), dtype=np.float64)
    for a, (_, indices) in enumerate(basins):
        W_a = np.sum(w[indices])
        if W_a > 0:
            chi[a, indices] = w_half[indices] / W_a
        else:
            chi[a, indices] = 0.0
            
    proj = chi @ eigenvectors_filtered
    weights = 1.0 / np.abs(eigenvalues_filtered)
    
    # Vectorized Mahalanobis distance calculation via cdist
    proj_scaled = proj * np.sqrt(weights)
    dist_matrix = cdist(proj_scaled, proj_scaled, metric='euclidean')
    
    # Scaling in accordance with the paper: \tilde{D} = RT * \sqrt{\nu_0} * D
    dist_matrix *= R_KCAL * temp_kelvin * np.sqrt(FREQUENCY_PREFACTOR)
    
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
    """Calculation of ultrametricity degrees."""
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
    Applied UNIFORMLY in the main stage and in all null models.
    """
    min_component_size = max(3, int(alpha_threshold * n_total_structures))
    
    global_basin_stats = []
    n_processed_components = 0
    n_significant_components = 0
    
    # For f_inter calculation: count triplets inside each component
    intra_triplets_total = 0
    
    for comp_idx, comp_indices in enumerate(graph_components):
        if len(comp_indices) < min_component_size:
            continue
            
        n_significant_components += 1
        comp_set = set(comp_indices)
        comp_energies = energies[list(comp_indices)]
        old_to_local = {old: local for local, old in enumerate(comp_indices)}
        
        # Build neighbor list inside the component
        comp_neighbors = []
        for old_idx in comp_indices:
            local_nbs = {old_to_local[nb] for nb in neighbors_list[old_idx] if nb in comp_set}
            comp_neighbors.append(local_nbs)
            
        # Find basins
        comp_basins_raw = compute_gradient_basins(comp_energies, comp_neighbors, verbose=False)
        
        # Filter basins
        RT = R_KCAL * temp_kelvin
        # FIXED: Shift by minimum energy to prevent overflow (Stage 9)
        min_e = np.min(comp_energies) if len(comp_energies) > 0 else 0.0
        Z = {i: sum(np.exp(-(comp_energies[idx] - min_e) / RT) for idx in indices)
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
                K_sym_comp, filtered_basins, comp_energies, num_requested, temp_kelvin,
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
    Performs double edge swap on a list of edges, strictly preserving vertex degrees.
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
    Control test: real spectrum, random basins of the same sizes.
    """
    _limit_threads_in_worker()
    (worker_id, N_comp, basin_sizes, eigenvalues_filtered, eigenvectors_filtered, comp_energies, temp_kelvin, eps, delta, seed) = args
    rng = np.random.RandomState(seed)
    
    indices = np.arange(N_comp)
    rng.shuffle(indices)
    
    random_basins = []
    start = 0
    for size in basin_sizes:
        random_basins.append((start, indices[start:start+size].tolist()))
        start += size
        
    K_basins = len(random_basins)
    RT = R_KCAL * temp_kelvin
    
    # FIXED: Apply Boltzmann distribution for random basins
    # subtracting minimum energy for numerical stability.
    min_e = np.min(comp_energies) if len(comp_energies) > 0 else 0.0
    w = np.exp(-(comp_energies - min_e) / RT)
    w_half = np.exp(-(comp_energies - min_e) / (2.0 * RT))
    
    chi = np.zeros((K_basins, N_comp), dtype=np.float64)
    for a, (_, indices_b) in enumerate(random_basins):
        W_a = np.sum(w[indices_b])
        if W_a > 0:
            chi[a, indices_b] = w_half[indices_b] / W_a
        else:
            chi[a, indices_b] = 0.0
            
    proj = chi @ eigenvectors_filtered
    weights = 1.0 / np.abs(eigenvalues_filtered)
    proj_scaled = proj * np.sqrt(weights)
    dist_matrix = cdist(proj_scaled, proj_scaled, metric='euclidean')
    dist_matrix *= R_KCAL * temp_kelvin * np.sqrt(FREQUENCY_PREFACTOR)
    
    u_nt, _, _, _ = compute_ultrametricity_score(dist_matrix, eps, delta)
    return u_nt

def _prepare_component_data_for_shuffle(neighbors_list, graph_components, alpha_threshold, n_total):
    """
    Preliminarily prepares data for all significant connected components.
    """
    min_component_size = max(3, int(alpha_threshold * n_total))
    components_data = []
    
    for comp_idx, comp_indices in enumerate(graph_components):
        if len(comp_indices) < min_component_size:
            continue
            
        comp_set = set(comp_indices)
        old_to_local = {old: local for local, old in enumerate(comp_indices)}
        
        comp_neighbors = []
        for old_idx in comp_indices:
            local_nbs = tuple(old_to_local[nb] for nb in neighbors_list[old_idx] if nb in comp_set)
            comp_neighbors.append(local_nbs)
            
        components_data.append((tuple(comp_indices), tuple(comp_neighbors), len(comp_indices)))
        
    return components_data

def _energy_shuffle_worker_optimized(args):
    """
    Optimized version of energy_shuffle.
    """
    _limit_threads_in_worker()
    (worker_id, real_energies, components_data, num_modes, temp_kelvin,
     min_basin_size, max_macrostates, gap_threshold, eigs_maxiter, eigs_sigma,
     eps, delta, seed) = args
     
    rng = np.random.RandomState(seed)
    t_start = time.time()
    
    try:
        shuffled_energies = rng.permutation(real_energies)
        t_perm = time.time() - t_start
        
        global_basin_stats = []
        intra_triplets_total = 0
        t_basins_total = 0.0
        t_ksym_total = 0.0
        t_spectral_total = 0.0
        n_components_processed = 0
        
        for comp_indices, comp_neighbors, n_comp in components_data:
            comp_energies = shuffled_energies[list(comp_indices)]
            
            t0 = time.time()
            comp_basins_raw = compute_gradient_basins(comp_energies, comp_neighbors, verbose=False)
            t_basins_total += time.time() - t0
            
            RT = R_KCAL * temp_kelvin
            # FIXED: Shift by minimum energy to prevent overflow (Stage 9)
            min_e = np.min(comp_energies) if len(comp_energies) > 0 else 0.0
            Z = {i: sum(np.exp(-(comp_energies[idx] - min_e) / RT) for idx in indices)
                 for i, (_, indices) in enumerate(comp_basins_raw)}
                 
            filtered_basins, _ = filter_macrostates_spectral(
                comp_basins_raw, Z, min_basin_size, max_macrostates, verbose=False
            )
            
            if len(filtered_basins) < 3:
                continue
                
            n_components_processed += 1
            
            t0 = time.time()
            K_sym_comp = build_transition_rate_matrix(comp_energies, comp_neighbors, temp_kelvin, FREQUENCY_PREFACTOR)
            t_ksym_total += time.time() - t0
            
            num_requested = min(num_modes, n_comp - 1)
            
            try:
                t0 = time.time()
                dist_matrix, evals_filt, evecs_filt, num_noise, num_phys = compute_spectral_distance(
                    K_sym_comp, filtered_basins, comp_energies, num_requested, temp_kelvin,
                    gap_threshold, eigs_maxiter, eigs_sigma, verbose=False
                )
                t_spectral_total += time.time() - t0
                
                u_nt_comp, u_tr_comp, u_non_comp, counts_comp = compute_ultrametricity_score(
                    dist_matrix, eps, delta
                )
                
                del dist_matrix, evals_filt, evecs_filt
                gc.collect()
            except Exception:
                continue
                
            del K_sym_comp
            
            n_triplets = sum(counts_comp.values())
            n_basins_comp = len(filtered_basins)
            intra_triplets_total += n_triplets
            
            global_basin_stats.append({
                'u_nt': u_nt_comp, 'u_tr': u_tr_comp, 'u_non': u_non_comp,
                'num_triplets': n_triplets, 'num_basins': n_basins_comp,
                'num_phys_modes': num_phys, 'num_noise_modes': num_noise
            })
            
        total_triplets = sum(s['num_triplets'] for s in global_basin_stats)
        if total_triplets == 0:
            return None
            
        weighted_u_nt = sum(s['u_nt'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
        weighted_u_tr = sum(s['u_tr'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
        weighted_u_non = sum(s['u_non'] * s['num_triplets'] for s in global_basin_stats) / total_triplets
        
        t_total = time.time() - t_start
        
        timing = {
            'perm': t_perm, 'basins': t_basins_total, 'ksym': t_ksym_total,
            'spectral': t_spectral_total, 'total': t_total,
            'n_comp_processed': n_components_processed
        }
        
        return (weighted_u_nt, weighted_u_tr, weighted_u_non, timing)
        
    except Exception:
        return None

def _config_model_worker(args):
    """
    Null hypothesis test (topo_shuffle / configuration model).
    """
    _limit_threads_in_worker()
    (worker_id, real_energies, edges_list, degree_sequence, graph_components_unused,
     n_total_structures, temp_kelvin, min_basin_size, max_macrostates, num_modes,
     gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold,
     n_swaps, seed) = args
     
    rng = np.random.RandomState(seed)
    t_start = time.time()
    
    try:
        t0 = time.time()
        shuffled_edges = _double_edge_swap(edges_list, n_swaps, rng)
        t_swap = time.time() - t0
        
        t0 = time.time()
        shuffled_energies = rng.permutation(real_energies)
        t_perm = time.time() - t0
        
        t0 = time.time()
        N = len(degree_sequence)
        new_neighbors_list = [set() for _ in range(N)]
        for u, v in shuffled_edges:
            new_neighbors_list[u].add(v)
            new_neighbors_list[v].add(u)
        t_build_graph = time.time() - t0
        
        t0 = time.time()
        new_components = find_connected_components(new_neighbors_list)
        t_find_comp = time.time() - t0
        
        t0 = time.time()
        result = _analyze_all_components(
            shuffled_energies, new_neighbors_list, new_components, n_total_structures,
            temp_kelvin, min_basin_size, max_macrostates, num_modes,
            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold
        )
        t_analysis = time.time() - t0
        
        del new_neighbors_list, new_components, shuffled_edges, shuffled_energies
        gc.collect()
        
        if result['total_triplets'] == 0:
            return None
            
        t_total = time.time() - t_start
        
        timing = {
            'swap': t_swap, 'perm': t_perm, 'build_graph': t_build_graph,
            'find_comp': t_find_comp, 'analysis': t_analysis, 'total': t_total,
            'n_comp_processed': result['n_processed_components']
        }
        
        return (result['weighted_u_nt'], result['weighted_u_tr'], result['weighted_u_non'], timing)
        
    except Exception:
        return None

def _nt_shuffle_worker(args):
    """
    Null hypothesis test (nt_shuffle).
    """
    _limit_threads_in_worker()
    (worker_id, original_seq, temp_kelvin, max_structures, energy_window,
     min_hairpin_len, min_basin_size, max_macrostates, num_modes,
     gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold, seed) = args
     
    rng = np.random.RandomState(seed)
    t_start = time.time()
    
    try:
        t0 = time.time()
        seq_list = list(original_seq)
        rng.shuffle(seq_list)
        shuffled_seq = ''.join(seq_list)
        t_shuffle = time.time() - t0
        
        t0 = time.time()
        structures, energies = generate_structures_stochastic(
            shuffled_seq, TEMPERATURE_CELSIUS, max_structures, energy_window, verbose=False
        )
        t_gen = time.time() - t0
        
        if len(structures) < 2:
            return None
            
        t0 = time.time()
        structures, energies = deduplicate_structures(structures, energies, verbose=False)
        t_dedup = time.time() - t0
        
        t0 = time.time()
        comp_map = {('A', 'U'): True, ('U', 'A'): True, ('G', 'C'): True,
                    ('C', 'G'): True, ('G', 'U'): True, ('U', 'G'): True}
        allowed_pairs, pair_to_idx, conflict_masks, bit, P = precompute_allowed_pairs_and_conflicts(
            len(shuffled_seq), shuffled_seq, min_hairpin_len, comp_map
        )
        t_precomp = time.time() - t0
        
        t0 = time.time()
        struct_masks = []
        struct_set_bits = []
        index_map = {}
        for idx, s in enumerate(structures):
            mask, set_bits = dotbracket_to_bitmask(s, pair_to_idx)
            struct_masks.append(mask)
            struct_set_bits.append(set_bits)
            index_map[mask] = idx
        t_bitmask = time.time() - t0
        
        t0 = time.time()
        neighbors = _build_neighbor_graph_local(
            struct_masks, struct_set_bits, index_map, conflict_masks, bit, P
        )
        t_build_graph = time.time() - t0
        
        del struct_masks, struct_set_bits, index_map
        gc.collect()
        
        t0 = time.time()
        graph_components = find_connected_components(neighbors)
        t_find_comp = time.time() - t0
        
        n_total = len(energies)
        
        t0 = time.time()
        result = _analyze_all_components(
            energies, neighbors, graph_components, n_total,
            temp_kelvin, min_basin_size, max_macrostates, num_modes,
            gap_threshold, eigs_maxiter, eigs_sigma, eps, delta, alpha_threshold
        )
        t_analysis = time.time() - t0
        
        del neighbors, graph_components, structures, energies
        gc.collect()
        
        if result['total_triplets'] == 0:
            return None
            
        t_total = time.time() - t_start
        
        timing = {
            'shuffle': t_shuffle, 'gen': t_gen, 'dedup': t_dedup, 'precomp': t_precomp,
            'bitmask': t_bitmask, 'build_graph': t_build_graph, 'find_comp': t_find_comp,
            'analysis': t_analysis, 'total': t_total,
            'n_comp_processed': result['n_processed_components']
        }
        
        return (result['weighted_u_nt'], result['weighted_u_tr'], result['weighted_u_non'], timing)
        
    except Exception:
        return None

def run_null_hypothesis_tests(real_u_nt, comp_res, temp_kelvin, n_workers, original_seq=None, verbose=False):
    """
    Launches the selected null hypothesis test depending on NULL_MODEL_TYPE.
    """
    if NULL_MODEL_TYPE == 'none':
        return None, None
        
    n_cpus = n_workers if n_workers else cpu_count()
    null_stats, control_stats = None, None
    
    run_energy = NULL_MODEL_TYPE in ('energy_shuffle', 'full_analysis')
    run_topo = NULL_MODEL_TYPE in ('topo_shuffle', 'full_analysis')
    run_nt = NULL_MODEL_TYPE == 'nt_shuffle'
    run_random = NULL_MODEL_TYPE == 'random_basins'
    
    # === RANDOM BASELINES TEST ===
    if run_random:
        print(f"\n--- CONTROL TEST: RANDOM BASINS (REAL SPECTRUM) ---")
        print(f"  Runs: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        
        basins = comp_res['basins']
        basin_sizes = [len(indices) for _, indices in basins]
        evals = comp_res['eigenvalues_filtered']
        evecs = comp_res['eigenvectors_filtered']
        N_comp = comp_res['comp_size']
        comp_energies = comp_res.get('comp_energies', np.zeros(N_comp))
        
        start_time = time.time()
        worker_args = [(i, N_comp, basin_sizes, evals, evecs, comp_energies, temp_kelvin,
                        ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, RANDOM_SEED + 20000 + i)
                       for i in range(NUM_NULL_SAMPLES)]
                       
        with Pool(processes=n_cpus, maxtasksperchild=1) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_random_basins_worker, worker_args)):
                results.append(res)
                if VERBOSE and NUM_NULL_SAMPLES > 1:
                    print(f"  Run {i+1} out of {NUM_NULL_SAMPLES} completed.", flush=True)
                    
        elapsed_test = time.time() - start_time
        u_nt_arr = np.array([r for r in results if r is not None])
        
        if len(u_nt_arr) > 0:
            mean_null = np.mean(u_nt_arr)
            observed_deviation = abs(real_u_nt - mean_null)
            p_value = float(np.mean(np.abs(u_nt_arr - mean_null) >= observed_deviation))
            
            control_stats = {
                'mean_u_nt': mean_null,
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'p_value': p_value,
                'elapsed': elapsed_test
            }
            
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:         {real_u_nt:.2f}%")
            print(f"  Random u_nt:       {control_stats['mean_u_nt']:.2f} +/- {control_stats['std_u_nt']:.2f}%")
            print(f"  p-value (two-sided): {control_stats['p_value']:.4f}")

    # === ENERGY SHUFFLE TEST (OPTIMIZED) ===
    energy_stats = None
    if run_energy:
        print(f"\n--- TEST 1: ENERGY SHUFFLE (GRAPH PRESERVED, OPTIMIZED) ---")
        print(f"  Runs: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Energy shuffling, processing ALL significant components)")
        
        real_energies = comp_res['all_energies']
        neighbors_list = comp_res['all_neighbors']
        graph_components = comp_res['all_components']
        n_total = comp_res['n_total_structures']
        num_modes = min(NUM_EIGENMODES, n_total - 1)
        
        print(f"  Preparing component data...")
        prep_start = time.time()
        components_data = _prepare_component_data_for_shuffle(
            neighbors_list, graph_components, ALPHA_COMPONENT_THRESHOLD, n_total
        )
        print(f"  Prepared {len(components_data)} significant components in {time.time()-prep_start:.1f} sec")
        
        start_time = time.time()
        worker_args = [
            (i, real_energies, components_data, num_modes, temp_kelvin,
             MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS,
             SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
             ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA,
             RANDOM_SEED + 30000 + i)
            for i in range(NUM_NULL_SAMPLES)
        ]
        
        with Pool(processes=n_cpus, maxtasksperchild=1) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_energy_shuffle_worker_optimized, worker_args, chunksize=1)):
                results.append(res)
                if VERBOSE and NUM_NULL_SAMPLES > 1:
                    print(f"  Run {i+1} out of {NUM_NULL_SAMPLES} completed.", flush=True)
                    
        elapsed_test = time.time() - start_time
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])
            
            mean_null = np.mean(u_nt_arr)
            observed_deviation = abs(real_u_nt - mean_null)
            p_value = float(np.mean(np.abs(u_nt_arr - mean_null) >= observed_deviation))
            
            energy_stats = {
                'mean_u_nt': mean_null,
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': p_value,
                'elapsed': elapsed_test
            }
            
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:         {real_u_nt:.2f}%")
            print(f"  EnergyShuffle u_nt:  {energy_stats['mean_u_nt']:.2f} +/- {energy_stats['std_u_nt']:.2f}%")
            print(f"  EnergyShuffle u_tr:  {energy_stats['mean_u_tr']:.2f} +/- {energy_stats['std_u_tr']:.2f}%")
            print(f"  EnergyShuffle u_non: {energy_stats['mean_u_non']:.2f} +/- {energy_stats['std_u_non']:.2f}%")
            print(f"  p-value (two-sided): {energy_stats['p_value']:.4f}")
            
            if verbose:
                timings = [r[3] for r in valid_results]
                avg_perm = np.mean([t['perm'] for t in timings])
                avg_basins = np.mean([t['basins'] for t in timings])
                avg_ksym = np.mean([t['ksym'] for t in timings])
                avg_spectral = np.mean([t['spectral'] for t in timings])
                avg_total = np.mean([t['total'] for t in timings])
                avg_n_comp = np.mean([t['n_comp_processed'] for t in timings])
                
                print(f"\n[TIME BREAKDOWN energy_shuffle (averaged over {len(timings)} runs)]")
                print(f"    Energy shuffling:      {avg_perm:.3f} sec")
                print(f"    Basin search:          {avg_basins:.3f} sec")
                print(f"    K_sym construction:    {avg_ksym:.3f} sec")
                print(f"    Spectral decomposition:{avg_spectral:.3f} sec")
                print(f"    Total per run:         {avg_total:.3f} sec")
                print(f"    Components processed:  {avg_n_comp:.1f} (on average)")
        else:
            print("  Failed to obtain any successful energy_shuffle run.")

    # === TOPO SHUFFLE TEST (configuration model) ===
    topo_stats = None
    if run_topo:
        print(f"\n--- TEST 2: CONFIGURATION MODEL + ENERGY SHUFFLE ---")
        print(f"  Runs: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Edge rewiring + energy shuffling, processing ALL significant components)")
        
        real_energies = comp_res['all_energies']
        neighbors_list = comp_res['all_neighbors']
        n_total = comp_res['n_total_structures']
        num_modes = min(NUM_EIGENMODES, n_total - 1)
        
        edges_set = set()
        for u, nbs in enumerate(neighbors_list):
            for v in nbs:
                if u < v:
                    edges_set.add((u, v))
        edges_list = list(edges_set)
        
        degree_sequence = np.zeros(n_total, dtype=int)
        for u, v in edges_list:
            degree_sequence[u] += 1
            degree_sequence[v] += 1
            
        n_swaps = NUM_EDGE_SWAPS_MULTIPLIER * len(edges_list)
        print(f"  Edges: {len(edges_list)}, Swaps per run: {n_swaps}")
        
        start_time = time.time()
        worker_args = [
            (i, real_energies, edges_list, degree_sequence, None, n_total,
             temp_kelvin, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, num_modes,
             SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
             ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
             n_swaps, RANDOM_SEED + 40000 + i)
            for i in range(NUM_NULL_SAMPLES)
        ]
        
        with Pool(processes=n_cpus, maxtasksperchild=1) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_config_model_worker, worker_args, chunksize=1)):
                results.append(res)
                if VERBOSE and NUM_NULL_SAMPLES > 1:
                    print(f"  Run {i+1} out of {NUM_NULL_SAMPLES} completed.", flush=True)
                    
        elapsed_test = time.time() - start_time
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])
            
            mean_null = np.mean(u_nt_arr)
            observed_deviation = abs(real_u_nt - mean_null)
            p_value = float(np.mean(np.abs(u_nt_arr - mean_null) >= observed_deviation))
            
            topo_stats = {
                'mean_u_nt': mean_null,
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': p_value,
                'elapsed': elapsed_test
            }
            
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:         {real_u_nt:.2f}%")
            print(f"  ConfigModel u_nt:    {topo_stats['mean_u_nt']:.2f} +/- {topo_stats['std_u_nt']:.2f}%")
            print(f"  ConfigModel u_tr:    {topo_stats['mean_u_tr']:.2f} +/- {topo_stats['std_u_tr']:.2f}%")
            print(f"  ConfigModel u_non:   {topo_stats['mean_u_non']:.2f} +/- {topo_stats['std_u_non']:.2f}%")
            print(f"  p-value (two-sided): {topo_stats['p_value']:.4f}")
            
            if verbose:
                timings = [r[3] for r in valid_results]
                avg_swap = np.mean([t['swap'] for t in timings])
                avg_perm = np.mean([t['perm'] for t in timings])
                avg_build = np.mean([t['build_graph'] for t in timings])
                avg_find = np.mean([t['find_comp'] for t in timings])
                avg_analysis = np.mean([t['analysis'] for t in timings])
                avg_total = np.mean([t['total'] for t in timings])
                avg_n_comp = np.mean([t['n_comp_processed'] for t in timings])
                
                print(f"\n[TIME BREAKDOWN topo_shuffle (averaged over {len(timings)} runs)]")
                print(f"    Edge rewiring:         {avg_swap:.3f} sec")
                print(f"    Energy shuffling:      {avg_perm:.3f} sec")
                print(f"    Graph construction:    {avg_build:.3f} sec")
                print(f"    Component search:      {avg_find:.3f} sec")
                print(f"    Component analysis:    {avg_analysis:.3f} sec")
                print(f"    Total per run:         {avg_total:.3f} sec")
                print(f"    Components processed:  {avg_n_comp:.1f} (on average)")
        else:
            print("  Failed to obtain any successful topo_shuffle run.")

    # === NT SHUFFLE TEST (nucleotide shuffling) ===
    nt_stats = None
    if run_nt:
        if original_seq is None:
            print(f"\nERROR: Original sequence is required for nt_shuffle mode.")
            return None, None
            
        print(f"\n--- TEST: NUCLEOTIDE SHUFFLE (FULL RECALCULATION) ---")
        print(f"  Runs: {NUM_NULL_SAMPLES}, Processes: {n_cpus}")
        print(f"  (Full regeneration + processing ALL significant components)")
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
        
        with Pool(processes=n_cpus, maxtasksperchild=1) as pool:
            results = []
            for i, res in enumerate(pool.imap_unordered(_nt_shuffle_worker, worker_args, chunksize=1)):
                results.append(res)
                if VERBOSE and NUM_NULL_SAMPLES > 1:
                    print(f"  Run {i+1} out of {NUM_NULL_SAMPLES} completed.", flush=True)
                    
        elapsed_test = time.time() - start_time
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) > 0:
            u_nt_arr = np.array([r[0] for r in valid_results])
            u_tr_arr = np.array([r[1] for r in valid_results])
            u_non_arr = np.array([r[2] for r in valid_results])
            
            mean_null = np.mean(u_nt_arr)
            observed_deviation = abs(real_u_nt - mean_null)
            p_value = float(np.mean(np.abs(u_nt_arr - mean_null) >= observed_deviation))
            
            nt_stats = {
                'mean_u_nt': mean_null,
                'std_u_nt': np.std(u_nt_arr, ddof=1) if len(u_nt_arr) > 1 else 0.0,
                'mean_u_tr': np.mean(u_tr_arr),
                'std_u_tr': np.std(u_tr_arr, ddof=1) if len(u_tr_arr) > 1 else 0.0,
                'mean_u_non': np.mean(u_non_arr),
                'std_u_non': np.std(u_non_arr, ddof=1) if len(u_non_arr) > 1 else 0.0,
                'p_value': p_value,
                'elapsed': elapsed_test
            }
            
            print(f"  Test completed in {elapsed_test:.1f} sec")
            print(f"  Real u_nt:         {real_u_nt:.2f}%")
            print(f"  NT-Shuffle u_nt:     {nt_stats['mean_u_nt']:.2f} +/- {nt_stats['std_u_nt']:.2f}%")
            print(f"  NT-Shuffle u_tr:     {nt_stats['mean_u_tr']:.2f} +/- {nt_stats['std_u_tr']:.2f}%")
            print(f"  NT-Shuffle u_non:    {nt_stats['mean_u_non']:.2f} +/- {nt_stats['std_u_non']:.2f}%")
            print(f"  p-value (two-sided): {nt_stats['p_value']:.4f}")
            
            if verbose:
                timings = [r[3] for r in valid_results]
                avg_shuffle = np.mean([t['shuffle'] for t in timings])
                avg_gen = np.mean([t['gen'] for t in timings])
                avg_dedup = np.mean([t['dedup'] for t in timings])
                avg_precomp = np.mean([t['precomp'] for t in timings])
                avg_bitmask = np.mean([t['bitmask'] for t in timings])
                avg_build = np.mean([t['build_graph'] for t in timings])
                avg_find = np.mean([t['find_comp'] for t in timings])
                avg_analysis = np.mean([t['analysis'] for t in timings])
                avg_total = np.mean([t['total'] for t in timings])
                avg_n_comp = np.mean([t['n_comp_processed'] for t in timings])
                
                print(f"\n[TIME BREAKDOWN nt_shuffle (averaged over {len(timings)} runs)]")
                print(f"    Nucleotide shuffling:    {avg_shuffle:.3f} sec")
                print(f"    Structure generation:  {avg_gen:.3f} sec")
                print(f"    Deduplication:         {avg_dedup:.3f} sec")
                print(f"    Pair precomputation:  {avg_precomp:.3f} sec")
                print(f"    Bitmask conversion:    {avg_bitmask:.3f} sec")
                print(f"    Graph construction:    {avg_build:.3f} sec")
                print(f"    Component search:      {avg_find:.3f} sec")
                print(f"    Component analysis:    {avg_analysis:.3f} sec")
                print(f"    Total per run:         {avg_total:.3f} sec")
                print(f"    Components processed:  {avg_n_comp:.1f} (on average)")
        else:
            print("  Failed to obtain any successful nt_shuffle run.")

    # === Forming a single result for full_analysis ===
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
    Scans the current folder for files with the .fasta extension.
    """
    try:
        from Bio import SeqIO
    except ImportError:
        print("Error: biopython package is required to work with FASTA files.")
        print("Install it with the command: pip install biopython")
        raise
        
    fasta_files = glob.glob("*.fasta")
    if not fasta_files:
        print("Error: no *.fasta files found in the current folder")
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
    print("Sequences (in ascending order of length):")
    for i, (seq, desc, length) in enumerate(all_sequences):
        print(f"  {i+1}. {desc}: length {length} nt")
        
    return all_sequences

def _single_stat_run(args):
    """
    Worker for parallel execution of a single NUM_STAT run.
    """
    _limit_threads_in_worker()
    (run_idx, seq, seq_description, seq_len, current_seed, show_details) = args
    
    t_total_start = time.time()
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
        
    neighbors = _build_neighbor_graph_local(
        struct_masks, struct_set_bits, index_map, conflict_masks, bit, P
    )
    
    del struct_masks, struct_set_bits, index_map
    gc.collect()
    
    graph_components = find_connected_components(neighbors)
    n_total = len(energies)
    
    analysis = _analyze_all_components(
        energies, neighbors, graph_components, n_total,
        temp_kelvin, MIN_MACROSTATE_SIZE, MAX_MACROSTATES_ANALYSIS, NUM_EIGENMODES,
        SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA,
        ULTRAMETRIC_EPSILON, ULTRAMETRIC_DELTA, ALPHA_COMPONENT_THRESHOLD,
        verbose=show_details
    )
    
    if analysis['total_triplets'] == 0:
        return None
        
    t_total = time.time() - t_total_start
    
    timing = {
        'precomp': time.time() - t_total_start,
        'gen': 0.0, 'dedup': 0.0, 'bitmask': 0.0,
        'build_graph': 0.0, 'find_comp': 0.0, 'analysis': 0.0,
        'total': t_total
    }
    
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
        'elapsed': t_total,
        'timing': timing,
        'all_energies': energies,
        'all_neighbors': neighbors,
        'all_components': graph_components,
        'n_total_structures': n_total,
        'original_seq': seq,
        'global_basin_stats': analysis['global_basin_stats']
    }

def process_sequence(seq, seq_description, seq_index, total_sequences):
    """
    Performs full ultrametricity analysis for a single RNA sequence.
    """
    seq_len = len(seq)
    n_workers = NUM_WORKERS if NUM_WORKERS else cpu_count()
    
    print("\n" + "=" * 70)
    print(f"PROCESSING SEQUENCE {seq_index} OUT OF {total_sequences}")
    print(f"Description: {seq_description}")
    print(f"Length: {seq_len} nucleotides")
    print(f"Seed: {RANDOM_SEED}")
    if NULL_MODEL_TYPE != 'none':
        print(f"NULL HYPOTHESIS TEST: {NULL_MODEL_TYPE} ({NUM_NULL_SAMPLES} runs)")
    print("=" * 70)
    
    main_start_time = time.time()
    
    worker_args = [
        (run_idx, seq, seq_description, seq_len, RANDOM_SEED + run_idx, VERBOSE)
        for run_idx in range(NUM_STAT)
    ]
    
    n_stat_workers = min(n_workers, NUM_STAT)
    
    if NUM_STAT > 1:
        print(f"\nPARALLEL LAUNCH OF {NUM_STAT} RUNS ON {n_stat_workers} PROCESSES...")
        with Pool(processes=n_stat_workers, maxtasksperchild=1) as pool:
            all_runs = pool.map(_single_stat_run, worker_args)
        all_runs = [r for r in all_runs if r is not None]
    else:
        result = _single_stat_run(worker_args[0])
        all_runs = [result] if result is not None else []
        
    main_elapsed = time.time() - main_start_time
    print(f"\n[LOG] Main stage time (NUM_STAT={NUM_STAT}): {main_elapsed:.2f} sec")
    
    if VERBOSE and all_runs:
        timings = [r['timing'] for r in all_runs if 'timing' in r]
        if timings:
            avg_total = np.mean([t['total'] for t in timings])
            print(f"\n[MAIN STAGE TIME BREAKDOWN (averaged over {len(timings)} runs)]")
            print(f"    Total per run:         {avg_total:.3f} sec")
            
    if not all_runs:
        print("  ERROR: no run completed successfully.")
        return None
        
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
    
    null_stats = None
    control_stats = None
    null_start_time = time.time()
    
    if NULL_MODEL_TYPE != 'none':
        print("\n" + "-" * 50)
        print("LAUNCHING NULL HYPOTHESIS TEST (based on 1st successful run)")
        print("-" * 50)
        
        temp_kelvin = TEMPERATURE_CELSIUS + 273.15
        original_seq_for_test = first_res.get('original_seq', seq)
        
        comp_res_for_null = {
            'all_energies': first_res['all_energies'],
            'all_neighbors': first_res['all_neighbors'],
            'all_components': first_res['all_components'],
            'n_total_structures': first_res['n_total_structures'],
            'basins': [],
            'comp_size': first_res['n_total_structures'],
            'eigenvalues_filtered': None,
            'eigenvectors_filtered': None,
            'comp_energies': None
        }
        
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
                min_e = np.min(comp_energies) if len(comp_energies) > 0 else 0.0
                Z = {i: sum(np.exp(-(comp_energies[idx] - min_e) / RT) for idx in indices)
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
                            K_sym_comp, filtered_basins, comp_energies, num_requested, temp_kelvin,
                            SPECTRAL_GAP_THRESHOLD, EIGS_MAXITER, EIGS_SIGMA, verbose=False
                        )
                        comp_res_for_null['basins'] = filtered_basins
                        comp_res_for_null['comp_size'] = len(main_comp)
                        comp_res_for_null['eigenvalues_filtered'] = evals
                        comp_res_for_null['eigenvectors_filtered'] = evecs
                        comp_res_for_null['comp_energies'] = comp_energies
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
            
    print("\n" + "=" * 70)
    print(f"STATISTICS OVER {len(all_runs)} RUNS")
    print("=" * 70)
    print(f"  Weighted u_nt:                {mean_u_nt:.2f} +/- {std_u_nt:.2f} %")
    print(f"  Weighted u_tr:                {mean_u_tr:.2f} +/- {std_u_tr:.2f} %")
    print(f"  Weighted u_non:               {mean_u_non:.2f} +/- {std_u_non:.2f} %")
    print(f"  Fraction of inter-component triplets: {f_inter:.4f} +/- {f_inter_std:.4f}")
    print(f"  Number of structures:         {mean_struct} +/- {std_struct}")
    print(f"  Number of basins:             {mean_basins} +/- {std_basins}")
    print(f"  Number of components (total): {mean_comp} +/- {std_comp}")
    print(f"  Execution time (main):        {mean_elapsed:.1f} +/- {std_elapsed:.1f} sec")
    
    if null_stats:
        if NULL_MODEL_TYPE == 'full_analysis':
            print(f"  --- FULL MECHANISM ANALYSIS ---")
            if null_stats.get('topo'):
                ts = null_stats['topo']
                print(f"  TopoShuffle (chaos) u_nt:     {ts['mean_u_nt']:.2f} +/- {ts['std_u_nt']:.2f} %")
                print(f"  TopoShuffle (chaos) u_tr:     {ts['mean_u_tr']:.2f} +/- {ts['std_u_tr']:.2f} %")
                print(f"  TopoShuffle (chaos) u_non:    {ts['mean_u_non']:.2f} +/- {ts['std_u_non']:.2f} %")
                print(f"  TopoShuffle p-value (two-sided): {ts['p_value']:.4f}")
            if null_stats.get('energy'):
                es = null_stats['energy']
                print(f"  EnergyShuffle (topology) u_nt:  {es['mean_u_nt']:.2f} +/- {es['std_u_nt']:.2f} %")
                print(f"  EnergyShuffle (topology) u_tr:  {es['mean_u_tr']:.2f} +/- {es['std_u_tr']:.2f} %")
                print(f"  EnergyShuffle (topology) u_non: {es['mean_u_non']:.2f} +/- {es['std_u_non']:.2f} %")
                print(f"  EnergyShuffle p-value (two-sided): {es['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'energy_shuffle':
            print(f"  --- Test: energy shuffle ---")
            print(f"  Shuffled u_nt:          {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
            print(f"  Shuffled u_tr:          {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
            print(f"  Shuffled u_non:         {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
            print(f"  p-value (two-sided):         {null_stats['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'topo_shuffle':
            print(f"  --- Test: configuration model ---")
            print(f"  Chaotic u_nt:           {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
            print(f"  Chaotic u_tr:           {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
            print(f"  Chaotic u_non:          {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
            print(f"  p-value (two-sided):         {null_stats['p_value']:.4f}")
        elif NULL_MODEL_TYPE == 'nt_shuffle':
            print(f"  --- Test: nucleotide shuffle ---")
            if null_stats:
                print(f"  NT-Shuffle u_nt:            {null_stats['mean_u_nt']:.2f} +/- {null_stats['std_u_nt']:.2f} %")
                print(f"  NT-Shuffle u_tr:            {null_stats['mean_u_tr']:.2f} +/- {null_stats['std_u_tr']:.2f} %")
                print(f"  NT-Shuffle u_non:           {null_stats['mean_u_non']:.2f} +/- {null_stats['std_u_non']:.2f} %")
                print(f"  p-value (two-sided):         {null_stats['p_value']:.4f}")
        else:
            print("  Results unavailable (see errors above)")
            
    if control_stats:
        print(f"  --- Control test (random basins) ---")
        print(f"  Random u_nt:             {control_stats['mean_u_nt']:.2f} +/- {control_stats['std_u_nt']:.2f} %")
        print(f"  p-value (two-sided):         {control_stats['p_value']:.4f}")
        
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
    elif null_stats and NULL_MODEL_TYPE == 'energy_shuffle':
        result_dict.update({
            'energy_p_value': null_stats['p_value'],
            'energy_u_nt_mean': null_stats['mean_u_nt'], 'energy_u_nt_std': null_stats['std_u_nt'],
            'energy_u_tr_mean': null_stats['mean_u_tr'], 'energy_u_tr_std': null_stats['std_u_tr'],
            'energy_u_non_mean': null_stats['mean_u_non'], 'energy_u_non_std': null_stats['std_u_non'],
            'energy_elapsed': null_stats['elapsed']
        })
    elif null_stats and NULL_MODEL_TYPE == 'topo_shuffle':
        result_dict.update({
            'topo_p_value': null_stats['p_value'],
            'topo_u_nt_mean': null_stats['mean_u_nt'], 'topo_u_nt_std': null_stats['std_u_nt'],
            'topo_u_tr_mean': null_stats['mean_u_tr'], 'topo_u_tr_std': null_stats['std_u_tr'],
            'topo_u_non_mean': null_stats['mean_u_non'], 'topo_u_non_std': null_stats['std_u_non'],
            'topo_elapsed': null_stats['elapsed']
        })
    elif null_stats and NULL_MODEL_TYPE == 'nt_shuffle':
        result_dict.update({
            'null_p_value_mean': null_stats['p_value'],
            'null_u_nt_mean': null_stats['mean_u_nt'], 'null_u_nt_std': null_stats['std_u_nt'],
            'null_u_tr_mean': null_stats['mean_u_tr'], 'null_u_tr_std': null_stats['std_u_tr'],
            'null_u_non_mean': null_stats['mean_u_non'], 'null_u_non_std': null_stats['std_u_non'],
            'null_elapsed': null_stats['elapsed']
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
    Main entry point of the program.
    """
    total_start_time = time.time()
    
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
    
    heavy_modes = ('energy_shuffle', 'topo_shuffle', 'full_analysis', 'nt_shuffle')
    if NULL_MODEL_TYPE in heavy_modes and NUM_NULL_SAMPLES > 10:
        print("=" * 70)
        print(f"WARNING: You have selected model '{NULL_MODEL_TYPE}' with a large number of runs.")
        print(f"Number of runs: {NUM_NULL_SAMPLES}")
        if NULL_MODEL_TYPE == 'nt_shuffle':
            print("The nt_shuffle mode requires FULL recalculation for each run.")
            print("This is VERY SLOW. It is recommended to set NUM_NULL_SAMPLES <= 5.")
        elif NULL_MODEL_TYPE == 'full_analysis':
            print("The full_analysis mode performs TWO heavy tests sequentially.")
            print("It is recommended to reduce NUM_NULL_SAMPLES to 20.")
        elif NULL_MODEL_TYPE == 'topo_shuffle':
            print("Each run requires graph rewiring, basin and spectrum recalculation.")
            print("It is recommended to reduce NUM_NULL_SAMPLES to 20.")
        else:
            print("Each run requires full recalculation of the transition matrix spectrum.")
            print("It is recommended to reduce NUM_NULL_SAMPLES to 20-30.")
        print("=" * 70)
        time.sleep(5)
        
    print("=" * 70)
    print("CALCULATION OF THE DEGREE OF NONTRIVIAL ULTRAMETRICITY")
    print("FOR MACROSTATES OF RNA SECONDARY STRUCTURE")
    print("METHOD: spectral Mahalanobis distance")
    print("(physically rigorous approach via transition rate matrix)")
    print("STRUCTURE GENERATION: stochastic sampling (pbacktrack)")
    if NUM_STAT > 1: print(f"STATISTICAL MODE: {NUM_STAT} runs (IN PARALLEL)")
    if NULL_MODEL_TYPE != 'none': print(f"NULL HYPOTHESIS TEST: {NULL_MODEL_TYPE} ({NUM_NULL_SAMPLES} runs, IN PARALLEL)")
    if NULL_MODEL_TYPE == 'none' and EXPECTATION_BY_RNA: print(f"EXPECTATION OVER RNA ENSEMBLE: ENABLED")
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
            
        if EXPECTATION_BY_RNA:
            n_res = len(all_seq_results)
            avg_len = np.mean([r['length'] for r in all_seq_results])
            u_nt_vals = np.array([r['weighted_u_nt'] for r in all_seq_results])
            u_tr_vals = np.array([r['weighted_u_tr'] for r in all_seq_results])
            u_non_vals = np.array([r['weighted_u_non'] for r in all_seq_results])
            f_int_vals = np.array([r['f_inter'] for r in all_seq_results])
            n_struct_vals = np.array([r['n_structures'] for r in all_seq_results], dtype=float)
            n_basin_vals = np.array([r['n_basins'] for r in all_seq_results], dtype=float)
            elapsed_vals = np.array([r['elapsed'] for r in all_seq_results])
            
            m_u_nt, s_u_nt = np.mean(u_nt_vals), np.std(u_nt_vals, ddof=1)
            m_u_tr, s_u_tr = np.mean(u_tr_vals), np.std(u_tr_vals, ddof=1)
            m_u_non, s_u_non = np.mean(u_non_vals), np.std(u_non_vals, ddof=1)
            m_f, s_f = np.mean(f_int_vals), np.std(f_int_vals, ddof=1)
            m_ns, s_ns = int(round(np.mean(n_struct_vals))), int(round(np.std(n_struct_vals, ddof=1)))
            m_nb, s_nb = int(round(np.mean(n_basin_vals))), int(round(np.std(n_basin_vals, ddof=1)))
            m_el, s_el = np.mean(elapsed_vals), np.std(elapsed_vals, ddof=1)
            
            avg_line = (f"{'':4} {'AVERAGE OVER ALL RNAs':<30} {avg_len:<8.0f} "
                        f"{m_ns}+/-{s_ns:<14} {m_nb}+/-{s_nb:<14} "
                        f"{m_f:.4f}+/-{s_f:.4f}{'':3} "
                        f"{m_u_nt:.2f}+/-{s_u_nt:.2f}{'':3} "
                        f"{m_u_tr:.2f}+/-{s_u_tr:.2f}{'':3} "
                        f"{m_u_non:.2f}+/-{s_u_non:.2f}{'':3} "
                        f"{m_el:.1f}+/-{s_el:.1f}")
            print(avg_line)
            
        if has_any_test:
            if has_energy:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: ENERGY SHUFFLE TEST (TOPOLOGY CONTRIBUTION)")
                print("=" * 70)
                header_e = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'Energy u_nt':<16} {'Energy u_tr':<16} {'Energy u_non':<16} {'p(Energy)':<10} {'Time (s)':<10}"
                print(header_e)
                print("-" * len(header_e))
                
                e_u_nt_vals, e_u_tr_vals, e_u_non_vals = [], [], []
                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"
                    
                    e_nt = res.get('energy_u_nt_mean')
                    e_tr = res.get('energy_u_tr_mean')
                    e_non = res.get('energy_u_non_mean')
                    e_pv = res.get('energy_p_value')
                    e_el = res.get('energy_elapsed')
                    
                    if e_nt is not None: e_u_nt_vals.append(e_nt)
                    if e_tr is not None: e_u_tr_vals.append(e_tr)
                    if e_non is not None: e_u_non_vals.append(e_non)
                    
                    e_nt_str = (f"{e_nt:.2f}+/-{res['energy_u_nt_std']:.2f}" if e_nt is not None else "N/A")
                    e_tr_str = (f"{e_tr:.2f}+/-{res['energy_u_tr_std']:.2f}" if e_tr is not None else "N/A")
                    e_non_str = (f"{e_non:.2f}+/-{res['energy_u_non_std']:.2f}" if e_non is not None else "N/A")
                    e_pv_str = (f"{e_pv:.4f}" if e_pv is not None else "N/A")
                    e_el_str = (f"{e_el:.1f}" if e_el is not None else "N/A")
                    
                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {e_nt_str:<16} {e_tr_str:<16} {e_non_str:<16} {e_pv_str:<10} {e_el_str:<10}"
                    print(line)
                    
                if EXPECTATION_BY_RNA and len(e_u_nt_vals) > 0:
                    m_ent = np.mean(e_u_nt_vals); s_ent = np.std(e_u_nt_vals, ddof=1)
                    m_etr = np.mean(e_u_tr_vals); s_etr = np.std(e_u_tr_vals, ddof=1)
                    m_enon = np.mean(e_u_non_vals); s_enon = np.std(e_u_non_vals, ddof=1)
                    avg_e_line = (f"{'':4} {'AVERAGE OVER ALL RNAs':<30} {'':8} {'':16} "
                                  f"{m_ent:.2f}+/-{s_ent:.2f}{'':<5} {m_etr:.2f}+/-{s_etr:.2f}{'':<5} "
                                  f"{m_enon:.2f}+/-{s_enon:.2f}{'':<5} {'':10} {'':10}")
                    print(avg_e_line)
                    
            if has_topo:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: TOPO SHUFFLE TEST (CONFIGURATION MODEL)")
                print("=" * 70)
                header_t = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'Topo u_nt':<16} {'Topo u_tr':<16} {'Topo u_non':<16} {'p(Topo)':<10} {'Time (s)':<10}"
                print(header_t)
                print("-" * len(header_t))
                
                t_u_nt_vals, t_u_tr_vals, t_u_non_vals = [], [], []
                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"
                    
                    t_nt = res.get('topo_u_nt_mean')
                    t_tr = res.get('topo_u_tr_mean')
                    t_non = res.get('topo_u_non_mean')
                    t_pv = res.get('topo_p_value')
                    t_el = res.get('topo_elapsed')
                    
                    if t_nt is not None: t_u_nt_vals.append(t_nt)
                    if t_tr is not None: t_u_tr_vals.append(t_tr)
                    if t_non is not None: t_u_non_vals.append(t_non)
                    
                    t_nt_str = (f"{t_nt:.2f}+/-{res['topo_u_nt_std']:.2f}" if t_nt is not None else "N/A")
                    t_tr_str = (f"{t_tr:.2f}+/-{res['topo_u_tr_std']:.2f}" if t_tr is not None else "N/A")
                    t_non_str = (f"{t_non:.2f}+/-{res['topo_u_non_std']:.2f}" if t_non is not None else "N/A")
                    t_pv_str = (f"{t_pv:.4f}" if t_pv is not None else "N/A")
                    t_el_str = (f"{t_el:.1f}" if t_el is not None else "N/A")
                    
                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {t_nt_str:<16} {t_tr_str:<16} {t_non_str:<16} {t_pv_str:<10} {t_el_str:<10}"
                    print(line)
                    
                if EXPECTATION_BY_RNA and len(t_u_nt_vals) > 0:
                    m_tnt = np.mean(t_u_nt_vals); s_tnt = np.std(t_u_nt_vals, ddof=1)
                    m_ttr = np.mean(t_u_tr_vals); s_ttr = np.std(t_u_tr_vals, ddof=1)
                    m_tnon = np.mean(t_u_non_vals); s_tnon = np.std(t_u_non_vals, ddof=1)
                    avg_t_line = (f"{'':4} {'AVERAGE OVER ALL RNAs':<30} {'':8} {'':16} "
                                  f"{m_tnt:.2f}+/-{s_tnt:.2f}{'':<5} {m_ttr:.2f}+/-{s_ttr:.2f}{'':<5} "
                                  f"{m_tnon:.2f}+/-{s_tnon:.2f}{'':<5} {'':10} {'':10}")
                    print(avg_t_line)
                    
            if has_nt:
                print("\n" + "=" * 70)
                print("FINAL SUMMARY REPORT: NT SHUFFLE TEST (NUCLEOTIDE SHUFFLING)")
                print("=" * 70)
                header_nt = f"{'No.':<4} {'Description':<30} {'Length':<8} {'Real u_nt':<16} {'NT u_nt':<16} {'NT u_tr':<16} {'NT u_non':<16} {'p(NT)':<10} {'Time (s)':<10}"
                print(header_nt)
                print("-" * len(header_nt))
                
                n_u_nt_vals, n_u_tr_vals, n_u_non_vals = [], [], []
                for i, res in enumerate(all_seq_results):
                    desc = res['description'][:28]
                    real_u = f"{res['weighted_u_nt']:.2f}"
                    if NUM_STAT > 1: real_u += f"+/-{res['weighted_u_nt_std']:.2f}"
                    
                    n_nt = res.get('null_u_nt_mean')
                    n_tr = res.get('null_u_tr_mean')
                    n_non = res.get('null_u_non_mean')
                    n_pv = res.get('null_p_value_mean')
                    n_el = res.get('null_elapsed')
                    
                    if n_nt is not None: n_u_nt_vals.append(n_nt)
                    if n_tr is not None: n_u_tr_vals.append(n_tr)
                    if n_non is not None: n_u_non_vals.append(n_non)
                    
                    n_nt_str = (f"{n_nt:.2f}+/-{res['null_u_nt_std']:.2f}" if n_nt is not None else "N/A")
                    n_tr_str = (f"{n_tr:.2f}+/-{res['null_u_tr_std']:.2f}" if n_tr is not None else "N/A")
                    n_non_str = (f"{n_non:.2f}+/-{res['null_u_non_std']:.2f}" if n_non is not None else "N/A")
                    n_pv_str = (f"{n_pv:.4f}" if n_pv is not None else "N/A")
                    n_el_str = (f"{n_el:.1f}" if n_el is not None else "N/A")
                    
                    line = f"{i+1:<4} {desc:<30} {res['length']:<8} {real_u:<16} {n_nt_str:<16} {n_tr_str:<16} {n_non_str:<16} {n_pv_str:<10} {n_el_str:<10}"
                    print(line)
                    
                if EXPECTATION_BY_RNA and len(n_u_nt_vals) > 0:
                    m_nnt = np.mean(n_u_nt_vals); s_nnt = np.std(n_u_nt_vals, ddof=1)
                    m_ntr = np.mean(n_u_tr_vals); s_ntr = np.std(n_u_tr_vals, ddof=1)
                    m_nnon = np.mean(n_u_non_vals); s_nnon = np.std(n_u_non_vals, ddof=1)
                    avg_n_line = (f"{'':4} {'AVERAGE OVER ALL RNAs':<30} {'':8} {'':16} "
                                  f"{m_nnt:.2f}+/-{s_nnt:.2f}{'':<5} {m_ntr:.2f}+/-{s_ntr:.2f}{'':<5} "
                                  f"{m_nnon:.2f}+/-{s_nnon:.2f}{'':<5} {'':10} {'':10}")
                    print(avg_n_line)
                    
        print(f"\nTotal execution time: {time.time() - total_start_time:.1f} seconds")

if __name__ == "__main__":
    main()
