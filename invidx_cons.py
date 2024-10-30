import json
import multiprocessing
import os
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from dict_cons import BPETokenizer  # Import tokenizer classes
from dict_cons import SimpleTokenizer, WordPieceTokenizer


def process_chunk(chunk,corpus_map,chunk_id,tokenizer, st_time):
    inverted_index = defaultdict(list)
    for i,doc_id  in enumerate(chunk):
        if time.time()-st_time>1260:
            break
        tokens = tokenizer.tokenize(corpus_map[doc_id])
        token_frequency = Counter(tokens)
        for token, frequency in token_frequency.items():
            inverted_index[token].append([doc_id , frequency])
    print(f"[DEBUG] Chunk {chunk_id} - {len(chunk)} documents")
    return inverted_index

def merge_inverted_index(results):
    merged_inverted_index = defaultdict(list)
    for inverted_index in results:
        for term, postings in inverted_index.items():
            merged_inverted_index[term].extend(postings)
    return merged_inverted_index

def build_inverted_index(corpus_map, tokenizer_type, vocab_size):
    start_time = time.time()
    merge_dict={}
    # make list of corpus items
    corpus = list(corpus_map.values())
    print(f"[DEBUG] Building inverted index with tokenizer type {tokenizer_type}")
    if tokenizer_type == 0:
        tokenizer = SimpleTokenizer()
    elif tokenizer_type == 1:
        tokenizer = BPETokenizer(vocab_size)
        tokenizer.fit(corpus, 280)
        merge_dict=tokenizer.merges
    elif tokenizer_type == 2:
        tokenizer = WordPieceTokenizer(vocab_size)
        tokenizer.fit(corpus, 280)
        merge_dict=tokenizer.merges
    n_cpus = os.cpu_count()
    corpus_items = list(corpus_map.keys())
    chunk_size = (len(corpus_items) + n_cpus -1) // n_cpus
    chunks = [corpus_items[i:i + chunk_size] for i in range(0, len(corpus_items), chunk_size)]
    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, chunk in enumerate(chunks):
            #  print size of each chunk
            print(f"[DEBUG] Chunk {idx} size: {len(chunk)}")
            futures.append(executor.submit(process_chunk, chunk, corpus_map,idx, tokenizer, start_time))
        results = [future.result() for future in as_completed(futures)]
    inverted_index = merge_inverted_index(results)
    print(f"[DEBUG] Inverted index built with {len(inverted_index)} terms.")
    return inverted_index, merge_dict


def parse_json_file(dirname):
    documents = {}
    print(f"[DEBUG] Starting to parse documents from {dirname}")
    st=0
    for filename in os.listdir(dirname):
        with open(os.path.join(dirname, filename), 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    id = data.get("doc_id", "")
                    title = data.get("title", "")
                    abstract = data.get("abstract", "")
                    doi = data.get("doi", "")
                    date = data.get("date", "")
                    combined_text = (title + " ")*2 + abstract
                    documents[id] = combined_text
                    st+=1
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Skipping line due to JSONDecodeError: {e}")
    print(f"[DEBUG] Parsed {len(documents)} documents.")
    #  print the size of list of cocno_toId

    return documents


def write_index_files(indexfile, inverted_index, tokenizer_type, merge_dict, total_documents):
    print(f"[DEBUG] Writing dictionary and index files to {indexfile}.dict and {indexfile}.idx")
    
    with open(f"{indexfile}.dict", "w", encoding="utf-8") as dict_file:
        for term in inverted_index.keys():
            dict_file.write(f"{term}\n")
    print(f"[DEBUG] Dictionary file {indexfile}.dict written.")
    #  first sort inverted index with respect to no. of postings in increaisng order
    # inverted_index = dict(sorted(inverted_index.items(), key=lambda x: len(x[1])))
    with open(f"{indexfile}.idx", "wb") as idx_file:
        idx_file.write(struct.pack('I', tokenizer_type))
        idx_file.write(struct.pack('I', total_documents))
        idx_file.write(struct.pack('I', len(merge_dict)))
        for pair, merge in merge_dict.items():
            pair_1_length = len(pair[0])
            idx_file.write(struct.pack('I', pair_1_length))
            idx_file.write(pair[0].encode('utf-8'))
            pair_2_length = len(pair[1])
            idx_file.write(struct.pack('I', pair_2_length))
            idx_file.write(pair[1].encode('utf-8'))
            merge_len=len(merge)
            idx_file.write(struct.pack('I', merge_len))
            idx_file.write(merge.encode('utf-8'))
        idx_file.write(struct.pack('I', len(inverted_index)))
        if tokenizer_type!=0:
            for term, postings in inverted_index.items():
                term_length = len(term)
                idx_file.write(struct.pack('I', term_length))
                idx_file.write(term.encode('utf-8'))
                postings_count = len(postings)
                idx_file.write(struct.pack('I', postings_count))
                for doc_id, frequency in postings:
                    idx_file.write(doc_id.encode('utf-8'))
                    idx_file.write(struct.pack('I', frequency))
        else:
            for term, postings in inverted_index.items():
                serialized_data = b''
                serialized_data += struct.pack('I', len(term))
                serialized_data += term.encode('utf-8')
                postings_count = len(postings)
                serialized_data += struct.pack('I', postings_count)
                for doc_id, frequency in postings:
                    serialized_data += doc_id.encode('utf-8') 
                    serialized_data += struct.pack('I', frequency)
                compressed_data = zlib.compress(serialized_data)
                compressed_data_length = len(compressed_data)
                idx_file.write(struct.pack('I', compressed_data_length))
                idx_file.write(compressed_data)
    print(f"[DEBUG] Index file {indexfile}.idx written.")

def main(json_file, indexfile, tokenizer_type):
    corpus= parse_json_file(json_dir)
    inverted_index, merge_dict = build_inverted_index(corpus, tokenizer_type, 5000)
    write_index_files(indexfile, inverted_index, tokenizer_type, merge_dict,len(corpus))
    print("[DEBUG] Inverted index creation completed.")

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) != 4:
        print("Usage: python invidx.py <json-file> <indexfile> <tokenizer_type>")
        print("tokenizer_type: 0 for Simple, 1 for BPE, 2 for WordPiece")
        sys.exit(1)
    json_dir = sys.argv[1]
    indexfile = sys.argv[2]
    tokenizer_type = int(sys.argv[3])
    main(json_dir, indexfile, tokenizer_type)
    print(f"[DEBUG] Execution time: {time.time() - start_time} seconds.")