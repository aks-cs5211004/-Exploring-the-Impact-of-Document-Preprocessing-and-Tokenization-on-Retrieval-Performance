#!/bin/bash

# Usage: invidx.sh [coll-path] [indexfile] {0|1|2}
# where
# [coll-path] specifies the path of the directory containing the documents
# [indexfile] is the base name for the index files
# {0|1|2} specifies the tokenizer type (0: Simple, 1: BPE, 2: WordPiece)

if [ $# -ne 3 ]; then
    echo "Usage: $0 [coll-path] [indexfile] {0|1|2}"
    exit 1
fi

COLL_PATH=$1
INDEXFILE=$2
TOKENIZER_TYPE=$3

# Run the inverted indexing Python script
python3 invidx_cons.py $COLL_PATH $INDEXFILE $TOKENIZER_TYPE