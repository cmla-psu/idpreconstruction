# Reconstruction Attacks on Aggressive Relaxations of Differential Privacy

This repository contains the supplementary materials (code and appendix with proofs) for the [Reconstruction Attacks on Aggressive Relaxations of Differential Privacy](full_manuscript/Reconstruction_against_aggresive_DP_relaxations.pdf).

## Dependencies
Python 3.6 or higher

## Running Experiments
To run the reconstruction experiments on a dataset protected using individual local sensitivity, run the IDPReconstruction.py script, which depends on the query answering mechanism in ProtectedDataset.py. The query-answering mechanism takes a filename (csv format), a k value, and lists of the categorical and numerical columns. This object can then be passed to the attacker class, along with a precision level, initial wide bounds for reconstructing numerical columns, and the order by which to reconstruct the columns. Reconstructed values are truncated, not rounded, to the precision value.

- NOTE: Only reconstruction for k=1 (i.e. individual differential privacy) is currently implemented. Implementing reconstruction for higher k values would require slight adjustments to the binarySearchValueReconstruction function and the reconstructCount function to search for a lower boundary between decimal and binary responses on high count values close to the number of rows in the database due to the limit this places on the sensitivity window.

To run the reconstruction experiments on an unprotected dataset, run the ZeroNoiseReconstruction.py script, which depends on the query answering mechanism in UnprotectedDataset.py.