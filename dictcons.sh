#!/bin/bash

# Usage: dictcons.sh [coll-path] {0|1|2}
# where
# [coll-path] specifies the path of the directory containing the training documents
# {0|1|2} specifies the tokenizer type (0: Simple, 1: BPE, 2: WordPiece)

if [ $# -ne 2 ]; then
    echo "Usage: $0 [coll-path] {0|1|2}"
    exit 1
fi

COLL_PATH=$1
TOKENIZER_TYPE=$2

# Run the tokenizer and dictionary construction Python script
python3 dict_cons.py $COLL_PATH $TOKENIZER_TYPE