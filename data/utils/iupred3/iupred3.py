#!/usr/bin/python3

import sys
import os
import iupred3_lib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("seqfile", help="FASTA formatted sequence file")
parser.add_argument("iupred_type", help="Analysis type: \"long\", \"short\" or \"glob\"")
parser.add_argument("-a", "--anchor", help="Enable ANCHOR2 prediction",
                    action="store_true")
parser.add_argument("-s", "--smoothing", help="Smoothing type: \"no\", \"medium\" or \"strong\". Default is \"medium\"",
                    default="medium")
# parser.add_argument("-e", "--experimental", help="Filter experimentally verified regions.",
#                     action="store_true")
args = parser.parse_args()

PATH = os.path.dirname(os.path.realpath(__file__))

if args.smoothing not in ['no', 'medium', 'strong']:
    raise ValueError('Smoothing (-s, --smoothing) must be either \"no\", \"medium\" or \"strong\"!')

sequence = iupred3_lib.read_seq(args.seqfile)
iupred2_result = iupred3_lib.iupred(sequence, args.iupred_type, smoothing=args.smoothing)
if args.anchor:
    anchor2_res = iupred3_lib.anchor2(sequence)
print("""# IUPred3 - improved prediction of protein disorder with a focus on specific user applications 
# Gábor Erdős, Mátyás Pajkos, Zsuzsanna Dosztányi
# Nucleic Acids Research 2021, Submitted
# 
# IUPred2A: context-dependent prediction of protein disorder as a function of redox state and protein binding
# Balint Meszaros, Gabor Erdos, Zsuzsanna Dosztanyi
# Nucleic Acids Research 2018;46(W1):W329-W337.
#
# Prediction type: {}
# Smoothing used: {}
# Prediction output""".format(args.iupred_type, args.smoothing))
if sys.argv[-1] == 'glob':
    print(iupred2_result[1])
print('# POS\tRES\tIUPRED2', end="\t")
if args.anchor:
    print("ANCHOR2", end='\t')
print()

for pos, residue in enumerate(sequence):
    print('{}\t{}\t{:.4f}'.format(pos + 1, residue, iupred2_result[0][pos]), end="")
    if args.anchor:
        print("\t{:.4f}".format(anchor2_res[pos]), end="")
    print()

