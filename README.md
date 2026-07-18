RNA KINETIC METRIC AND ULTRAMETRICITY ANALYSIS

1. OVERVIEW

This repository provides a comprehensive Python implementation for calculating the degree of nontrivial ultrametricity of RNA secondary structure energy landscapes. The method is based on a physically rigorous kinetic metric constructed via the spectral decomposition of the symmetrized Kramers transition rate matrix and the Mahalanobis distance between attraction basins. 

The software accompanies the scientific paper "Kinetic metric for attraction basins of RNA secondary structures and analysis of the energy landscape ultrametricity" by A. P. Zubarev. It is designed to process ensembles of RNA secondary structures, build transition graphs, identify gradient basins, compute spectral distances, and rigorously test the statistical significance of the observed hierarchical organization against multiple null models.


2. THEORETICAL BACKGROUND

Hierarchical organization of complex systems is naturally described by ultrametric spaces, where the strong triangle inequality holds. For RNA molecules, the energy landscape of secondary structures may exhibit partial ultrametricity, reflecting the presence of thermodynamic funnels and hierarchical folding pathways.

Previous methods, such as the single-linkage distance, artificially enforce ultrametricity by construction, making it impossible to distinguish between real physical hierarchy and methodological artifacts. 

Our method constructs a kinetic metric that is NOT an ultrametric by construction. It models the folding dynamics as a continuous-time Markov process. The transition rates between neighboring structures are calculated using the Kramers formula. The asymmetric transition rate matrix is symmetrized using the detailed balance condition. The distance between macroscopic attraction basins is then defined as the Mahalanobis distance in the subspace spanned by the slowest relaxation modes (eigenvectors corresponding to the smallest non-zero eigenvalues). This approach accounts for all possible transition paths, thermodynamic weights, and high energy barriers.


3. KEY FEATURES

- Physically Rigorous Approach: Distance between basins accounts for all possible transition paths and thermodynamic weights via spectral decomposition.
- Automatic Noise Filtering: Separates physical relaxation modes from numerical noise by detecting a spectral gap.
- Disconnected Graphs Handling: RNA structure graphs generated via stochastic sampling are often disconnected. The software analyzes all significant connected components separately and computes a weighted average of the ultrametricity scores.
- Comprehensive Null Models: Includes Energy Shuffle, Topology Shuffle, Nucleotide Shuffle, and Random Basins models to assess statistical significance via two-sided p-values.
- High Performance: Fully parallelized execution for both the main analysis and null models using Python multiprocessing.
- Bitmask Optimization: Uses O(1) bitwise conflict checking and highly optimized Inter-Process Communication (IPC) for rapid neighbor generation.
- Detailed Diagnostics: Provides verbose logging with precise time breakdowns for every computational stage.


4. DEPENDENCIES AND INSTALLATION

The code requires Python 3 and the following external packages:

- ViennaRNA package (with Python bindings)
- NumPy
- SciPy
- Biopython
- threadpoolctl

Installation Steps:

Step 1: Install the ViennaRNA package.
On Linux (Debian/Ubuntu): sudo apt-get install python3-viennarna
Using Conda (Cross-platform): conda install -c bioconda viennarna

Step 2: Install the Python dependencies using pip:
pip install numpy scipy biopython threadpoolctl

Note on Windows: The script includes specific thread-limiting routines (via threadpoolctl and environment variables) to prevent BLAS/OpenMP thread oversubscription, which is a common issue with multiprocessing on Windows.


5. USAGE AND DATA INPUT

The program is executed directly from the command line:
python rna_ultrametricity.py

The script supports two data input modes, controlled by the FASTA_RNA parameter at the top of the script:

- FASTA MODE (FASTA_RNA = True): The script scans the current working directory for all files with the .fasta extension. It loads all sequences, filters out invalid characters, and processes them in ascending order of length.
- SINGLE SEQUENCE MODE (FASTA_RNA = False): The script processes a single RNA sequence hardcoded in the RNA_SEQUENCE variable.


6. DETAILED PARAMETER GUIDE

All user-configurable parameters are located in the "USER PARAMETERS" section at the top of the script.

6.1. Temperature and Energy Parameters
- TEMPERATURE_CELSIUS: Temperature in Celsius for Boltzmann weights and transition rates (default 37.0).
- ENERGY_WINDOW: Energy cutoff relative to the minimum free energy (MFE) in kcal/mol. Structures with energy greater than MFE + ENERGY_WINDOW are discarded. Set to "inf" to disable the cutoff.

6.2. Structure Generation Parameters
- MAX_STRUCTURES: Maximum number of unique secondary structures to sample via stochastic backtrack (pbacktrack).
- MIN_HAIRPIN_LEN: Minimum number of unpaired nucleotides in a hairpin loop (steric constraint, default 3).
- RANDOM_SEED: Seed for the random number generator to ensure reproducibility.

6.3. Attraction Basins and Components
- MAX_MACROSTATES_ANALYSIS: Maximum number of attraction basins kept for the final spectral analysis. If more remain after filtering, basins with the highest partition functions are retained.
- MIN_MACROSTATE_SIZE: Minimum number of structures a basin must contain to be considered statistically significant.
- ALPHA_COMPONENT_THRESHOLD: Relative threshold for classifying connected components as significant or noise. A component is significant if it contains at least max(3, ALPHA_COMPONENT_THRESHOLD x N) structures, where N is the total number of unique structures.

6.4. Spectral Analysis Parameters
- NUM_EIGENMODES: Number of eigenmodes requested for spectral decomposition using the Lanczos method.
- SPECTRAL_GAP_THRESHOLD: Threshold for detecting a spectral gap. If the ratio of consecutive sorted eigenvalues exceeds this value, smaller modes are discarded as numerical noise.
- FREQUENCY_PREFACTOR: Prefactor in the Kramers formula. Affects the absolute scale of the matrix but cancels out in relative distance calculations.
- EIGS_MAXITER: Maximum iterations for the ARPACK Lanczos algorithm.
- EIGS_SIGMA: Shift-invert parameter for finding eigenvalues near zero.

6.5. Ultrametricity Testing Parameters
- ULTRAMETRIC_EPSILON: Relative precision for considering the two largest sides of a triangle as equal.
- ULTRAMETRIC_DELTA: Minimum relative difference between the smaller and middle sides to classify a triangle as nontrivially ultrametric.
- EPS_COMPARISON: Threshold for comparing real numbers to handle floating-point inaccuracies (e.g., identifying plateaus).

6.6. Computational and Statistical Parameters
- NUM_WORKERS: Number of parallel processes. Recommended to be less than the number of physical CPU cores.
- VERBOSE: Toggles detailed logging, including time breakdowns for each computational stage.
- NUM_STAT: Number of independent statistical runs per sequence. Results are averaged and reported as mean +/- standard deviation.
- EXPECTATION_BY_RNA: If True, adds a summary line with the ensemble average across all processed sequences.


7. NULL HYPOTHESIS TESTING FRAMEWORK

A central feature of this software is the ability to test whether the observed ultrametricity is a specific biological property or a generic feature of random heteropolymers. The NULL_MODEL_TYPE parameter selects the testing framework. All tests use a two-sided p-value to detect both unusually high and unusually low ultrametricity.

7.1. Energy Shuffle (energy_shuffle)
Preserves the exact neighborhood graph (including all topological correlations) but randomly shuffles the free energies of the structures. Basins and the spectrum are recalculated. This tests whether the hierarchical organization is driven by specific thermodynamic funnels rather than just the graph topology.

7.2. Topology Shuffle (topo_shuffle)
Implements the Configuration Model. It rewires graph edges using double-edge swaps (strictly preserving the degree sequence of every vertex) and shuffles the energies. This destroys topological correlations while maintaining the mobility distribution, establishing a baseline chaos level for random graphs.

7.3. Nucleotide Shuffle (nt_shuffle)
The strictest biological control. It randomly permutes the original RNA sequence while preserving its exact nucleotide composition. For each permutation, the entire pipeline is re-executed: structure generation, graph building, component analysis, and spectral decomposition. This tests if the specific evolutionary order of nucleotides is responsible for the landscape hierarchy.

7.4. Random Basins (random_basins)
A geometric control test. It preserves the real spectrum of the transition matrix but randomly partitions the structures into basins of the exact same sizes. This checks whether the observed ultrametricity is merely an artifact of the high-dimensional eigenvector space geometry.

7.5. Full Analysis (full_analysis)
Automatically executes both the Energy Shuffle and Topology Shuffle tests and outputs separate summary tables for each.


8. OUTPUT INTERPRETATION

The script outputs detailed logs to the console and generates final summary tables. Key metrics include:

- u_nt (Nontrivial Ultrametricity): The percentage of triplets satisfying the strong triangle inequality with distinct sides. This is the primary measure of hierarchical organization.
- u_tr (Trivial Ultrametricity): The percentage of equilateral triangles (all three distances are equal).
- u_non (Non-ultrametric): The percentage of triplets violating the ultrametric condition.
- f_inter (Fragmentation Index): The fraction of triplets that belong to different disconnected components. A high f_inter indicates that the landscape is highly fragmented at the given energy window, meaning the global hierarchy is broken into isolated local hierarchies.
- p-value: The two-sided statistical significance of the deviation from the null model ensemble. Values less than 0.05 indicate that the real RNA sequence exhibits a statistically anomalous degree of ultrametricity compared to the randomized models.


9. ALGORITHMIC OPTIMIZATIONS

To handle the combinatorial explosion of RNA secondary structures, the software employs several advanced optimizations:

- Bitmask IPC: Structures are encoded as bitmasks of allowed base pairs. Conflict checking (e.g., pseudoknots or overlapping pairs) is performed in O(1) time using bitwise AND operations.
- Path Compression: Gradient descent for basin identification uses an iterative path-compression algorithm instead of recursion, preventing stack overflow on large plateaus.
- Shift-Invert Spectral Decomposition: Uses the shift-invert mode of the ARPACK library to rapidly find the smallest non-zero eigenvalues of the sparse, negative semi-definite transition matrix.
- Pre-allocated Component Data: For the Energy Shuffle null model, component topology is serialized and passed to worker processes only once, drastically reducing Inter-Process Communication overhead.
- Thread Limiting: Explicitly limits BLAS/LAPACK/OpenMP threads inside multiprocessing workers to prevent CPU thrashing and memory leaks on both Windows and Linux.


10. CITATION

If you use this software or the underlying methodology in your research, please cite the accompanying paper:

A. P. Zubarev, "Kinetic metric for attraction basins of RNA secondary structures and analysis of the energy landscape ultrametricity", 2026.


11. LICENSE

Please refer to the LICENSE file in the repository for terms of use and distribution.
