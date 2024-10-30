import json
import math
import os
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

from dict_cons import BPETokenizer, SimpleTokenizer, WordPieceTokenizer


def load_index_files(indexfile, dictfile):
    terms = []
    doc_id_set=set()
    inverted_index = defaultdict(list)
    merge_dict = {}
    total_documents=0
    dict_file_path = dictfile
    if not os.path.isfile(dict_file_path):
        raise FileNotFoundError(f"[ERROR] Dictionary file {dict_file_path} not found.")
    with open(dict_file_path, "r", encoding="utf-8") as dict_file:
        terms = [line.strip() for line in dict_file.readlines()]
    print(f"[DEBUG] Dictionary file {dict_file_path} read with {len(terms)} terms.")
    idx_file_path = indexfile
    if not os.path.isfile(idx_file_path):
        raise FileNotFoundError(f"[ERROR] Index file {idx_file_path} not found.")  
    idx_file_size = os.path.getsize(idx_file_path)
    print(f"[DEBUG] Index file {idx_file_path} size: {idx_file_size} bytes")
    st=0
    with open(idx_file_path, "rb") as idx_file:
        tokenizer_type = struct.unpack('I', idx_file.read(4))[0]
        print(f"[DEBUG] Tokenizer type: {tokenizer_type}")
        total_documents = struct.unpack('I', idx_file.read(4))[0]
        print(f"[DEBUG] Total documents: {total_documents}")
        merge_len = struct.unpack('I', idx_file.read(4))[0]
        print(f"[DEBUG] Merge length: {merge_len}")
        for _ in range(merge_len):
            pair_1_length = struct.unpack('I', idx_file.read(4))[0]
            pair_1 = idx_file.read(pair_1_length).decode('utf-8')
            pair_2_length = struct.unpack('I', idx_file.read(4))[0]
            pair_2 = idx_file.read(pair_2_length).decode('utf-8')
            merge_len = struct.unpack('I', idx_file.read(4))[0]
            merge = idx_file.read(merge_len).decode('utf-8')
            merge_dict[(pair_1, pair_2)] = merge
            print(f"[DEBUG] Loaded merge for pair '{pair_1}', '{pair_2}' -> {merge}")
        inverted_index_len = struct.unpack('I', idx_file.read(4))[0]
        print(f"[DEBUG] Inverted index length: {inverted_index_len}")
        if(tokenizer_type==0):
            inverted_index_len= inverted_index_len
        if tokenizer_type == 0:
            for _ in range(inverted_index_len):
                compressed_data_length = struct.unpack('I', idx_file.read(4))[0]
                compressed_data = idx_file.read(compressed_data_length)
                serialized_data = zlib.decompress(compressed_data)
                offset = 0
                term_length = struct.unpack('I', serialized_data[offset:offset + 4])[0]
                offset += 4
                term = serialized_data[offset:offset + term_length].decode('utf-8')
                offset += term_length
                postings_count = struct.unpack('I', serialized_data[offset:offset + 4])[0]
                offset += 4
                postings = []
                for _ in range(postings_count):
                    doc_id = serialized_data[offset:offset + 8].decode('utf-8')
                    doc_id_set.add(doc_id)
                    offset += 8
                    frequency = struct.unpack('I', serialized_data[offset:offset + 4])[0]
                    offset += 4
                    postings.append((doc_id, frequency))

                inverted_index[term] = postings
                st+=1
        else:
            for _ in range(inverted_index_len):
                term_length = struct.unpack('I', idx_file.read(4))[0]
                term = idx_file.read(term_length).decode('utf-8')
                if term not in terms:
                    print(f"[ERROR] Term '{term}' not found in dictionary file.")
                    continue
                postings_count = struct.unpack('I', idx_file.read(4))[0]
                postings = []
                for _ in range(postings_count):
                    doc_id = idx_file.read(8).decode('utf-8')
                    doc_id_set.add(doc_id)
                    frequency = struct.unpack('I', idx_file.read(4))[0]
                    postings.append((doc_id, frequency))
                    
                inverted_index[term] = postings
        print(f"[DEBUG] Loaded inverted index with {len(inverted_index)} terms.")
    return inverted_index, tokenizer_type, total_documents, merge_dict, doc_id_set

def tokenize_query(query, merge_dict, tokenizer_type):
    if tokenizer_type == 0:
        tokenizer = SimpleTokenizer()
        return tokenizer.tokenize(query)
    elif tokenizer_type == 1:
        tokenizer = BPETokenizer(5000)
    elif tokenizer_type == 2:
        tokenizer = WordPieceTokenizer(5000)
    STokenizer = SimpleTokenizer()
    pre_tokenized_text = STokenizer.tokenize(query)
    splits = [[l for l in word if word.isascii()] for word in pre_tokenized_text]
    for idx, split in enumerate(splits):
        for pair, merge in merge_dict.items():
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])

def search(query, inverted_index, merge_dict, doc_ids, idf, tfidf_matrix, tokenizer_type, doc_norms):
    print(f"[DEBUG] Processing query: '{query}'")
    query_terms = tokenize_query(query, merge_dict, tokenizer_type)
    print(f"[DEBUG] Tokenized query terms: {query_terms}")
    query_tfidf = defaultdict(float)
    # make a conter over query terms
    query_term_freq = Counter(query_terms)
    for term, freq in query_term_freq.items():
        if term in idf:
            tf = 1 + math.log2(freq)
            query_tfidf[term] = tf * idf[term]
    print(f"[DEBUG] Query TF-IDF vector: {dict(query_tfidf)}")
    scores = {}
    query_norm = math.sqrt(sum(weight**2 for weight in query_tfidf.values()))
    for doc_id in doc_ids:
        common_terms = set(query_tfidf.keys()) & set(tfidf_matrix[doc_id].keys())
        dot_product = sum(query_tfidf[term] *  tfidf_matrix[doc_id][term] for term in common_terms)
        scores[doc_id] = dot_product / (doc_norms[doc_id] * query_norm)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scores

def process_queries(queryfile, inverted_index, merge_dict, total_documents, tokenizer_type, resultfile, doc_id_set):
    
    # Calculate IDF
    idf = {term: math.log2(1 + len(doc_id_set) / len(postings)) for term, postings in inverted_index.items()}
    
    # Calculate TF-IDF matrix
    tfidf_matrix = defaultdict(dict)
    for term, postings in inverted_index.items():
        for doc_id, freq in postings:
            tf = 1 + math.log2(freq)
            tfidf_matrix[doc_id][term] = tf * idf[term]
    
    # Calculate document norms
    doc_norms = {doc_id: math.sqrt(sum(weight**2 for weight in tfidf_matrix[doc_id].values())) for doc_id in doc_id_set}
    
    # Process queries and measure time
    query_times = []
    with open(queryfile, 'r', encoding='utf-8') as file:
        queries = [json.loads(line.strip()) for line in file]
    with open(resultfile, 'w', encoding='utf-8') as file:
        for query_data in queries:
            query_id = query_data.get("query_id")
            description = query_data.get("description", "")
            title = query_data.get("title", "")
            query = description + " " + (title + " ")*2
            
            # Measure the time for each query
            start_time = time.time()
            results = search(query, inverted_index, merge_dict, doc_id_set, idf, tfidf_matrix, tokenizer_type, doc_norms)
            end_time = time.time()
            
            # Record query time
            query_time = end_time - start_time
            query_times.append((query_id, query_time))
            
            # Write results to file
            for doc_id, score in results[:100]:
                file.write(f"{query_id} 0 {doc_id} {score:.20f}\n")
            print(f"[DEBUG] Query '{query_id}' processed in {query_time:.4f} seconds")
    
    # Plot query times
    plot_query_times(query_times)
    
    total_time = sum(query_time for _, query_time in query_times)
    num_queries = len(queries)
    avg_time_per_query = total_time / num_queries if num_queries > 0 else 0
    
    print(f"[DEBUG] Query processing completed in {total_time:.2f} seconds.")
    print(f"[DEBUG] Average time per query: {avg_time_per_query:.2f} seconds.")
    print(f"[DEBUG] Results written to {resultfile}")

def plot_query_times(query_times):
    query_ids, times = zip(*query_times)
    plt.figure(figsize=(10, 6))
    plt.plot(query_ids, times, marker='o', linestyle='-')
    plt.xlabel('Query ID')
    plt.ylabel('Time (seconds)')
    plt.title('Query Processing Time per Query')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def main(queryfile, resultfile, indexfile, dictfile):
    inverted_index, tokenizer_type, total_documents, merge_dict, doc_id_set= load_index_files(indexfile, dictfile)
    print("[DEBUG] Index files loaded.")
    process_queries(queryfile, inverted_index, merge_dict, total_documents, tokenizer_type, resultfile, doc_id_set)
    print("[DEBUG] Query processing completed.")

if __name__ == "__main__":
    start_time=time.time()
    if len(sys.argv) != 5:
        print("Usage: python tfidf_search.py <queryfile> <resultfile> <indexfile> <dictfile>")
        sys.exit(1)
    
    queryfile = sys.argv[1]
    resultfile = sys.argv[2]
    indexfile = sys.argv[3]
    dictfile = sys.argv[4]
    main(queryfile, resultfile, indexfile, dictfile)
    print(f"[DEBUG] Execution time: {time.time() - start_time} seconds.")
