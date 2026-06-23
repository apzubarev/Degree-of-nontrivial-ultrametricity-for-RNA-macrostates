INSTRUCTIONS FOR USING THE RNA ENERGY LANDSCAPE ULTRAMETRICITY ANALYSIS SOFTWARE

1. GENERAL INFORMATION

This software implements a method for calculating the degree of nontrivial ultrametricity of the energy landscape of RNA secondary structures. The method is based on the spectral decomposition of the symmetrized Kramers transition rate matrix and the Mahalanobis distance. Unlike single-linkage distance methods, which produce an ultrametric by construction, this metric does not automatically satisfy the ultrametric inequality, making the verification of hierarchical landscape organization a substantive mathematical problem rather than an algorithmic artifact.

The software implements a complete computational pipeline, including:
- Stochastic sampling of secondary structures from the Boltzmann ensemble.
- Construction of the adjacency graph via elementary operations.
- Automatic filtering of spurious spectral modes using a spectral gap search.
- Processing of disconnected structure graphs arising from stochastic sampling.
- A hierarchy of null models for statistical significance testing (energy shuffle, configuration model, nucleotide shuffle).
- Statistical significance assessment via a two-sided p-value, as biological function may require either pronounced hierarchy (high ultrametricity) or its absence/specific frustration (low ultrametricity).
- Parallel execution of the main stage, statistical runs (NUM_STAT), and null hypothesis tests.

2. REQUIREMENTS AND INSTALLATION

Python version 3.8 or higher is required.

Required libraries:
- ViennaRNA (Python bindings)
- numpy
- scipy
- biopython
- threadpoolctl (optional, for explicit thread control on Windows)

Install dependencies via pip:
pip install numpy scipy biopython threadpoolctl

The ViennaRNA library is recommended to be installed via conda:
conda install -c bioconda viennarna

Note for Windows users: The software includes critical fixes for multiprocessing with the spawn start method, limiting BLAS/OpenMP threads inside workers to prevent resource contention and memory leaks.

3. PARAMETER CONFIGURATION

All parameters are set in the "USER PARAMETERS" block at the beginning of the script file. Key parameters include:

DATA SOURCE
-----------
FASTA_RNA (True/False)
Data loading mode. True reads all .fasta files in the current directory, sorts them by length, and processes all sequences. False uses the sequence defined in the RNA_SEQUENCE variable. Default: True.

RNA_SEQUENCE (string)
Primary RNA structure. Used only when FASTA_RNA = False. Valid characters: A, U, G, C (uppercase, T is automatically replaced by U). Default: "ACATCAATCCACCACTCTTTCTCTTTAAAAAGAGTAGACCCAGGAACCGAAATTCTTTACCAAATTAAAAAA".

TEMPERATURE AND ENERGY
----------------------
TEMPERATURE_CELSIUS (number)
Calculation temperature in degrees Celsius. Affects Boltzmann weights and transition probabilities. Default: 37.0 (physiological temperature). Valid range: 0.0 – 100.0.

ENERGY_WINDOW (number or "inf")
Energy window relative to the minimum free energy (MFE) in kcal/mol. Structures with energy above MFE + ENERGY_WINDOW are discarded. "inf" disables the window. Default: 50.0.

STRUCTURE GENERATION
--------------------
MAX_STRUCTURES (integer)
Maximum sample size of unique structures. Generation stops when this limit is reached. Default: 100000. Valid range: 100 – 20000 (higher values possible but increase computation time).

MIN_HAIRPIN_LEN (integer)
Minimum number of unpaired nucleotides in a hairpin loop. Standard value: 3 (steric constraint). Default: 3. Valid range: 0 – 10.

RANDOM_SEED (integer)
Initial value for the random number generator. Ensures reproducibility. When NUM_STAT > 1, seeds vary as RANDOM_SEED, RANDOM_SEED+1, ..., RANDOM_SEED+NUM_STAT-1. Default: 43.

ATTRACTION BASINS
-----------------
MAX_MACROSTATES_ANALYSIS (integer)
Maximum number of attraction basins participating in the final analysis per connected component. If more remain after filtering, basins with the highest partition functions Z are kept. Default: 500. Valid range: 3 – 500.

MIN_MACROSTATE_SIZE (integer)
Minimum attraction basin size (number of constituent structures). Basins smaller than this are considered statistically insignificant and are excluded. Default: 5. Valid range: 1 – 100.

ALPHA_COMPONENT_THRESHOLD (float)
Relative threshold for classifying connected components as significant or noise. A component is significant if it contains at least max(3, ALPHA_COMPONENT_THRESHOLD * N) structures, where N is the total number of unique structures. Noise components are excluded from f_inter calculation and the final spectral analysis. Default: 0.001. Valid range: 0.001 – 0.1.

SPECTRAL ANALYSIS
-----------------
NUM_EIGENMODES (integer)
Number of eigenmodes requested for spectral decomposition. After automatic noise filtering, the actual number used may be smaller. Must be strictly less than the number of structures. Default: 50. Valid range: 5 – 200.

SPECTRAL_GAP_THRESHOLD (float)
Threshold for detecting the spectral gap between noise and physical modes. If the ratio |lambda_k| / |lambda_{k-1}| exceeds this value, modes with smaller indices are discarded as numerical noise. Default: 1e6. Valid range: 1e2 – 1e12.

FREQUENCY_PREFACTOR (float)
Frequency prefactor nu_0 in the Kramers formula. Affects absolute scale but not relative distances. Default: 1.0.

EIGS_MAXITER (integer)
Maximum number of iterations for the Lanczos algorithm (ARPACK). Increase for matrices with a dense spectrum near zero. Default: 50000. Valid range: 1000 – 200000.

EIGS_SIGMA (float)
Shift sigma for the shift-invert Lanczos method. Optimal value (1e-4 to 1e-5) ensures diagonal dominance and fast LU factorization. Default: 1e-4. Valid range: 1e-6 – 1e-3.

ULTRAMETRICITY CHECK
--------------------
ULTRAMETRIC_EPSILON (float)
Relative precision epsilon for approximate ultrametricity. Two largest triangle sides are considered equal if (d_max - d_mid) / d_mid <= epsilon. Must be strictly less than ULTRAMETRIC_DELTA. Default: 0.05. Valid range: 0.0 – 0.20.

ULTRAMETRIC_DELTA (float)
Minimum relative difference delta for nontrivial ultrametric classification: (d_mid - d_min) / d_mid > delta. Must be strictly greater than ULTRAMETRIC_EPSILON. Default: 0.1. Valid range: 0.01 – 0.50.

NUMERICAL PRECISION
-------------------
EPS_COMPARISON (float)
Threshold for comparing real numbers (energies, distances). Used for strict inequalities in plateau, local minima, and triangle classification. Default: 1e-9. Valid range: 1e-12 – 1e-6.

COMPUTATIONAL RESOURCES
-----------------------
NUM_WORKERS (integer or None)
Number of parallel processes for neighbor generation, NUM_STAT runs, and null models. On Windows, recommended to set below the number of physical cores (e.g., 12 for a 24-thread processor). None auto-detects all cores. Default: 12.

VERBOSE (True/False)
Verbose output mode. True outputs all intermediate results and detailed timing breakdowns. False outputs only final results. Default: True.

STATISTICAL ANALYSIS
--------------------
NUM_STAT (integer)
Number of independent runs for each RNA sequence. Runs are executed in parallel. Results are averaged and output as mean +/- std. Integer quantities are rounded. Default: 5. Valid range: 1 – 100.

NULL HYPOTHESIS TESTING
-----------------------
NULL_MODEL_TYPE (string)
Type of null model for statistical testing. All models use a two-sided p-value. Possible values:
- none: No null hypothesis testing. The program operates in normal mode.
- full_analysis: (RECOMMENDED) Full mechanism analysis. Automatically executes both energy_shuffle and topo_shuffle tests. Outputs two separate summary tables.
- energy_shuffle: Preserves graph topology while shuffling energies. Assesses the contribution of pure graph topology to ultrametricity. Optimized via pre-computation of component data.
- topo_shuffle: Configuration model. Edge rewiring (double_edge_swap preserving vertex degrees) combined with energy shuffling. Destroys topological correlations while preserving mobility distribution. Represents maximum chaos at a fixed degree sequence.
- nt_shuffle: Complete nucleotide shuffling with full ensemble regeneration. The most stringent biological control. Tests whether ultrametricity is determined by specific nucleotide order. Requires significant computation time.
- random_basins: Geometric control. Preserves real spectrum but randomly partitions structures into basins of identical sizes. Checks for artifacts of high-dimensional eigenvector space geometry.
Default: 'energy_shuffle'.

NUM_NULL_SAMPLES (integer)
Number of null model realizations. Executed in parallel via multiprocessing.Pool. For random_basins, 100-500 is feasible. For energy_shuffle and topo_shuffle, 20-30 is recommended. For nt_shuffle, 5-10 is recommended. In full_analysis mode, this number applies to both tests. Default: 100.

NUM_EDGE_SWAPS_MULTIPLIER (integer)
Multiplier for edge swaps in topo_shuffle and full_analysis modes. Number of swaps = NUM_EDGE_SWAPS_MULTIPLIER * |E|. Higher values ensure thorough destruction of topological correlations. Default: 10. Valid range: 1 – 100.

ENSEMBLE AVERAGING
------------------
EXPECTATION_BY_RNA (True/False)
Ensemble averaging mode. When True, a summary row "AVERAGE OVER ALL RNAs" containing mean values and standard deviations of all corresponding metrics is appended to the END OF EACH output table (main results table and all null model tables). This allows assessment of typical values and spread across the ensemble for both real data and each null model. Default: False.

4. RUNNING THE SOFTWARE

Execute from the command line:
python Vienna_RNA_new_ver_23.py

Or run from an interactive environment (Jupyter, Spyder, IDLE).

Before starting calculations, the program outputs all configured parameters, warns about resource-intensive configurations (large NUM_NULL_SAMPLES with heavy test modes), lists the found FASTA files, and displays the loaded sequences sorted by length.

If NUM_STAT > 1, all runs for each sequence are executed in parallel. The null hypothesis test (if enabled) is executed once per sequence using data from the first successful run.

5. INTERPRETATION OF RESULTS

5.1. Main Results Table

The main table contains weighted average values across all significant connected components. The weights are the number of basin triplets within each component, ensuring that larger components contribute proportionally more to the final metric.

Columns:
- No.: Sequence number.
- Description: Sequence identifier (from FASTA header).
- Length: Sequence length in nucleotides.
- Structures: Total number of unique structures in the sample (mean +/- std over NUM_STAT runs).
- Basins: Total number of significant basins summed over ALL connected components. Note: MAX_MACROSTATES_ANALYSIS applies to each component individually, so the total may exceed this limit when multiple significant components are present.
- f_inter: Fraction of inter-component triplets. An indicator of graph fragmentation. Ranges from 0 (all basins in one component) to values close to 1 (basins distributed across many isolated components). When f_inter > 0.8, u_nt characterizes only local hierarchy within individual components rather than the global landscape structure.
- u_nt (%): Degree of nontrivial ultrametricity. Characterizes the presence of hierarchical organization where two larger distances are approximately equal and significantly larger than the smallest. Values range from 0 to 100 percent.
- u_tr (%): Degree of trivial ultrametricity (equilateral triangles). Typically close to zero. High values would indicate a lack of differentiated hierarchy.
- u_non (%): Degree of non-ultrametricity. Triangles not satisfying either ultrametricity condition.
- Time (s): Execution time for the main stage.

5.2. Null Model Tables

When tests are enabled (NULL_MODEL_TYPE != 'none'), additional tables are displayed. For full_analysis mode, two separate tables are produced: Energy Shuffle and Topo Shuffle (Configuration Model).

Each null model table includes:
- Real u_nt from the main analysis.
- Mean and std of u_nt, u_tr, u_non for the null model ensemble.
- Two-sided p-value indicating whether the observed ultrametricity significantly deviates from the null model.
- Test execution time.

Interpretation of two-sided p-value:
- p < 0.05: The observed ultrametricity significantly deviates from the random level (in either direction). The hierarchy (or its absence) is a statistically significant property of the given sequence.
- p >= 0.05: Ultrametricity does not significantly differ from the random model.

Model comparison logic:
- If real differs significantly from nt_shuffle but not from energy_shuffle: Ultrametricity is a topological property of the conformational space, not a consequence of specific nucleotide order.
- If real significantly exceeds energy_shuffle: The specific distribution of free energies (thermodynamic funnels) enhances ultrametricity beyond what topology alone provides.
- If real is significantly lower than energy_shuffle: Physical organization frustrates the topologically inherent hierarchy.
- If real is approximately equal to topo_shuffle: The observed ultrametricity is consistent with a random graph having the same degree distribution, suggesting it arises from generic constraints rather than specific organization.

5.3. Ensemble Averaging (EXPECTATION_BY_RNA)

When EXPECTATION_BY_RNA = True, an additional summary row is appended to every output table:
- Main Results Table: Mean and std of u_nt, u_tr, u_non, f_inter, number of structures, basins, components, and execution time across all RNA sequences.
- Energy Shuffle Table: Mean and std of Energy u_nt, Energy u_tr, Energy u_non across all sequences.
- Topo Shuffle Table: Mean and std of Topo u_nt, Topo u_tr, Topo u_non across all sequences.
- NT Shuffle Table: Mean and std of NT u_nt, NT u_tr, NT u_non across all sequences.

This row provides a quick overview of the typical ultrametricity level and its variability across the studied ensemble, both for real landscapes and under each null hypothesis. It is particularly useful for identifying whether observed patterns are consistent across the dataset or dominated by individual outliers.

6. COMPUTATIONAL PERFORMANCE

The most resource-intensive stages are neighbor graph construction (Stage 5) and spectral decomposition (Stage 12). The software is optimized using bit masks for O(1) conflict checking and parallel computing via multiprocessing.

Key optimizations:
- Bitmask-based structure representation enables O(1) conflict checking during neighbor generation.
- Parallel neighbor generation distributes work across multiple processes.
- The energy_shuffle null model uses pre-computed component data to avoid repeated serialization of the full adjacency graph.
- Shift-invert Lanczos method with sigma = EIGS_SIGMA ensures fast spectral decomposition even for random graphs.
- maxtasksperchild=1 prevents memory leaks from C extensions (SuperLU, ARPACK) in long-lived processes.

Approximate timing for an RNA of 70 nucleotides with a sample of ~22,000 structures: one main stage run takes 20-30 seconds on a modern multi-core processor (12 workers). The energy_shuffle test with 20 realizations takes several minutes. The nt_shuffle test requires complete recalculation for each realization and may take hours depending on NUM_NULL_SAMPLES.

7. CITATION

If you use this software in your research, please cite:
Zubarev, A. P. (2026). Degree of nontrivial ultrametricity for RNA macrostates. Zenodo. https://doi.org/10.5281/zenodo.20818030

8. CONTACTS

Author: A. P. Zubarev
Email: apzubarev@mail.ru
