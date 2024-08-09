Phase separation propensity predictor, returns a single score per sequence contained within a fasta file, ignoring sequences that are shorter than any sequence in the training/test set or that have ambigious residues (outside the scope of the training process).

This predictor was designed primarily with the goal of testing the relationship between planar protein-protein contacts and phase separation, but validation of the predictor demonstrates the ability to separate known phase separating proteins from other proteomic sequences.

For use in predicting sequences note that the contact predictions and validation tests both derive primarily from sequences found in nature, and the supported use of this program is in highlighting full sequences as found in proteomic datasets. Scores for artificial sequences and for arbitrary segments of proteins represent untested extrapolations.

DEPENDENCIES:

requires the numpy python library
database load times are faster with python vernon >= 3.5

USAGE:

   ./elife_phase_separation_predictor.py [fasta file]

   command line options
         -residue_scores
            (writes out weighted per-residue scores)
         -score_components
            (writes out the eight unweighted per-residue score components)
         -output [filename]
            (writes to specified file instead of to the screen)
         -overwrite
            (output will overwrite an existing file)
         -mute
            (output [filename] mode will be silent)

By default the script has to be run from the directory containing the database folder ("./DBS/")


