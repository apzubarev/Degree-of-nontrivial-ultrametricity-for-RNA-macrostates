INSTRUCTIONS FOR USING THE RNA ENERGY LANDSCAPE ULTRAMETRICITY ANALYSIS SOFTWARE

1. GENERAL INFORMATION

This software implements a method for calculating the degree of nontrivial ultrametricity of the energy landscape of RNA secondary structures. The method is based on the spectral decomposition of the symmetrized Kramers transition rate matrix and the Mahalanobis distance. Unlike single-linkage distance methods, which produce an ultrametric by construction, this metric does not automatically satisfy the ultrametric inequality, making the verification of hierarchical landscape organization a substantive mathematical problem rather than an algorithmic artifact.

The software implements a complete computational pipeline, including automatic filtering of spurious spectral modes, processing of disconnected structure graphs arising from stochastic sampling, and a hierarchy of null models for statistical significance testing.

2. REQUIREMENTS AND INSTALLATION

Python version 3.8 or higher is required.

Required libraries:
- ViennaRNA (Python bindings)
- numpy
- scipy
- biopython

Install dependencies via pip:
pip install numpy scipy biopython

The ViennaRNA library is recommended to be installed via conda:
conda install -c bioconda viennarna

3. PARAMETER CONFIGURATION

All parameters are set in the "USER PARAMETERS" block at the beginning of the script file. Key parameters include:

FASTA_RNA (True/False)
Data loading mode. True reads all .fasta files in the current directory. False uses the sequence defined in the RNA_SEQUENCE variable.

TEMPERATURE_CELSIUS (number)
Calculation temperature in degrees Celsius. Default value: 37.0.

ENERGY_WINDOW (number)
Energy window relative to the minimum free energy (MFE) in kcal/mol. Structures with energy above MFE + ENERGY_WINDOW are discarded. Default value: 50.0.

MAX_STRUCTURES (integer)
Maximum sample size of unique structures. Default value: 100000.

NUM_STAT (integer)
Number of independent runs for reproducibility assessment. Runs are executed in parallel. Default value: 1. A value of 3-5 is recommended for estimating standard deviation.

NULL_MODEL_TYPE (string)
Type of null model for statistical testing. Possible values:
- none: no null hypothesis testing
- energy_shuffle: shuffling energies while preserving graph topology
- topo_shuffle: configuration model (edge rewiring + energy shuffling)
- nt_shuffle: complete nucleotide shuffling with ensemble regeneration
- full_analysis: automatic execution of both energy_shuffle and topo_shuffle
- random_basins: geometric control (random basins of the same sizes)

NUM_NULL_SAMPLES (integer)
Number of null model realizations. For nt_shuffle, 5-10 is recommended; for other models, 20-30.

NUM_WORKERS (integer or None)
Number of parallel processes. None enables automatic detection based on available CPU cores.

4. RUNNING THE SOFTWARE

Execute from the command line:
python Vienna_RNA_new_ver_21.py

Or run from an interactive environment (Jupyter, Spyder, IDLE).

Before starting calculations, the program outputs all configured parameters, the list of found FASTA files, and the loaded sequences.

5. INTERPRETATION OF RESULTS

5.1. Main Results Table

Contains weighted average values across all significant connected components:

u_nt (%) — degree of nontrivial ultrametricity. Characterizes the presence of hierarchical organization. Values range from 0 to 100 percent.

u_tr (%) — degree of trivial ultrametricity. Typically close to zero.

u_non (%) — degree of non-ultrametricity.

f_inter — fraction of inter-component triplets. An indicator of graph fragmentation. When f_inter > 0.8, the u_nt value characterizes only local hierarchy within individual components rather than the global landscape.

5.2. Null Model Tables

When tests are enabled, additional tables with p-values and test execution times are displayed.

Interpretation of p-value:
- p < 0.05: the observed ultrametricity significantly exceeds the random level. The hierarchy is due to specific landscape organization.
- p >= 0.05: ultrametricity does not differ from the random model.

Model comparison:
- If real > nt_shuffle but real is approximately equal to energy_shuffle: hierarchy is determined by the topology of the conformational space.
- If real > energy_shuffle: hierarchy is determined by the specific distribution of free energies (thermodynamic funnels).

6. COMPUTATIONAL PERFORMANCE

The most resource-intensive stages are neighbor graph construction and spectral decomposition. The software is optimized using bit masks and parallel computing.

Approximate timing for an RNA of 70 nucleotides with a sample of 5000 structures: one main stage run takes 20-30 seconds on a modern multi-core processor. The nt_shuffle test requires complete recalculation and takes proportionally longer.

7. CITATION

If you use this software in your research, please cite:
Zubarev, A. P. (2026). Degree of nontrivial ultrametricity for RNA macrostates. Zenodo. https://doi.org/10.5281/zenodo.20611640

8. CONTACTS

Author: A. P. Zubarev
Email: apzubarev@mail.ru
