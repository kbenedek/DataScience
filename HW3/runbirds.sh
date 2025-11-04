#!/bin/bash

# 1. Feature extraction: Convert the bird data CSV to transactional format (space separated attributes)
# Use birds2025ext.csv as the input file. The output is birdstrans.txt.
python3 featex.py

# 2. Convert attribute labels to numerical codes: Create birdstrans.txt.codes and the translation table birdtable.txt.
./namescodes/namescodes -n birdstrans.txt -tbirdtable.txt -L

# 3. Convert constraint labels to numerical codes: Use the same translation table.
# ./namescodes/namescodes -n bconstr.txt -tbirdtable.txt

# 4. Search association rules with Kingfisher: Use the coded input data and coded constraints.
# Parameters used here are illustrative examples: -k170 (upper bound for item number), -M-5 (initial threshold), -q300 (top 300 rules).
./kingfisher/kingfisher -i birdstrans.txt.codes -k300 -M-5 -q300 -o birdrules.txt # -b bconstr.txt.codes

# 5. Convert numerical codes back to label names: Transform the rule results using the table.
./namescodes/namescodes -c birdrules.txt -tbirdtable.txt