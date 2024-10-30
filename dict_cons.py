import gzip
import json
import math
import multiprocessing
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Pool

from tokenizers import Tokenizer
from tokenizers.models import WordPiece as WordPieceModel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


class SimpleTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\s+|[,.:;"’]+')
    def tokenize(self, text):
        tokens = [token for token in self.pattern.split(text) if token.isascii()]
        return tokens

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = []
        self.start_time = None
        self.word_freqs = defaultdict(int)
        self.merges = {}
        self.splits= {}
    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for a, b in zip(split, split[1:]):
                pair_freqs[(a, b)] += freq
        return pair_freqs
    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    def process_chunk(self, texts):
        local_word_freqs = defaultdict(int)
        local_vocab = set()
        tokenizer = SimpleTokenizer()
        for text in texts:
            words = tokenizer.tokenize(text)
            for word in words:
                local_word_freqs[word] += 1
                for letter in word:
                    local_vocab.add(letter)
        return local_word_freqs, local_vocab
    def fit(self, corpus, tim=287):
        self.start_time = time.time()
        n_cpus = multiprocessing.cpu_count()
        chunk_size = len(corpus) // n_cpus
        chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            for future in futures:
                local_word_freqs, local_vocab = future.result()
                for word, freq in local_word_freqs.items():
                    self.word_freqs[word] += freq
                self.vocab.extend(local_vocab)
        self.vocab = sorted(set(self.vocab))
        self.splits = {
            word: [c for c in word]
            for word in self.word_freqs.keys()
        }
        print(f"[DEBUG] Completed INIT fit method in {time.time() - self.start_time:.2f} seconds.")
        while len(self.vocab) < self.vocab_size:
            if time.time() - self.start_time > tim:
                print("[DEBUG] Time limit reached. Stopping the merge process.")
                break
            pair_freqs = self.compute_pair_freqs()
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if(best_pair == ""):
                print("[DEBUG] No pair found. Stopping the merge process.")
                break
            self.merge_pair(*best_pair)
            print(f"[DEBUG] Merging pair {best_pair} with frequency {max_freq}")
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
            
        # print stats ans sizes
        print(f"[DEBUG] Vocab size: {len(self.vocab)}")
        print(f"[DEBUG] Merges size: {len(self.merges)}")
        
            
    def tokenize(self,text):
        Stoken=SimpleTokenizer()
        pre_tokenized_text = Stoken.tokenize(text)
        splits = [[l for l in word] for word in pre_tokenized_text]
        for idx, split in enumerate(splits):
            for pair, merge in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

class WordPieceTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = []
        self.start_time = None
        self.word_freqs = defaultdict(int)
        self.merges = {}
        self.splits= {}
    def compute_pair_freqs(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split =self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for letter in split:
                letter_freqs[letter] += freq
            for a, b in zip(split, split[1:]):
                pair_freqs[(a, b)] += freq
        scores = {
            pair: (freq) / math.log2(letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores
    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits
    def process_chunk(self, texts):
        local_word_freqs = defaultdict(int)
        local_vocab = set()
        tokenizer = SimpleTokenizer()
        for text in texts:
            words = tokenizer.tokenize(text)
            for word in words:
                local_word_freqs[word] += 1
                for letter in word:
                    local_vocab.add(letter)
        return local_word_freqs, local_vocab
    def fit(self, corpus, tim=287):
        self.start_time = time.time()
        n_cpus = multiprocessing.cpu_count()
        chunk_size = len(corpus) // n_cpus
        chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            for future in futures:
                local_word_freqs, local_vocab = future.result()
                for word, freq in local_word_freqs.items():
                    self.word_freqs[word] += freq
                self.vocab.extend(local_vocab)
        self.vocab = sorted(set(self.vocab))
        self.splits = {
            word: [c for c in word]
            for word in self.word_freqs.keys()
        }
        print(f"[DEBUG] Completed INIT fit method in {time.time() - self.start_time:.2f} seconds.")
        while len(self.vocab) < self.vocab_size:
            if time.time() - self.start_time > tim:
                print("[DEBUG] Time limit reached. Stopping the merge process.")
                break
            pair_freqs = self.compute_pair_freqs()
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            if(best_pair == ""):
                print("[DEBUG] No pair found. Stopping the merge process.")
                break
            self.merge_pair(*best_pair)
            print(f"[DEBUG] Merging pair {best_pair} with frequency {max_freq}")
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])   
        # print stats ans sizes
        print(f"[DEBUG] Vocab size: {len(self.vocab)}")
        print(f"[DEBUG] Merges size: {len(self.merges)}")
          
    def tokenize(self,text):
        Stoken=SimpleTokenizer()
        pre_tokenized_text = Stoken.tokenize(text)
        splits = [[l for l in word] for word in pre_tokenized_text]
        for idx, split in enumerate(splits):
            for pair, merge in self.merges.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

def parse_json_file(dirname):
    documents = []
    #  iterate over all files of dir
    for filename in os.listdir(dirname):
        with open(os.path.join(dirname, filename), 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    title = data.get("title", "")
                    abstract = data.get("abstract", "")
                    doi = data.get("doi", "")
                    date = data.get("date", "")
                    combined_text = f"{title} {abstract} {doi} {date}"
                    documents.append(combined_text)
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] Skipping line due to JSONDecodeError: {e}")
    print(f"[DEBUG] Parsed {len(documents)} documents.")
    return documents

def main(dir_path, tokenizer_type):
    documents = parse_json_file(dir_path)
    sample_size = (len(documents) +1) // 2
    sampled_documents = documents[:sample_size]
    if tokenizer_type == 0:
        tokenizer = SimpleTokenizer()
        tokens = []
        for doc in documents:
            tokens.extend(tokenizer.tokenize(doc))
        print(f"[DEBUG] Found {len(set(tokens))} unique tokens.")
        with open("output.dict", "w") as out_file:
            for word in set(tokens):
                out_file.write(word + "\n")
            print("[DEBUG] Simple tokens written to output_simple.dict.")
    elif tokenizer_type == 1:
        tokenizer = BPETokenizer(50000)
        tokenizer.fit(sampled_documents, 293)
        with open("output.dict", "w") as out_file:
            for word in tokenizer.vocab:
                out_file.write(word + "\n")
            print("[DEBUG] BPE tokens written to output_bpe.dict.")
    elif tokenizer_type == 2:
        tokenizer = WordPieceTokenizer(50000)
        tokenizer.fit(sampled_documents, 293)
        with open("output.dict", "w") as out_file:
            for word in tokenizer.vocab:
                out_file.write(word + "\n")
            print("[DEBUG] WordPiece tokens written to output_wordpiece.dict.")
    elif tokenizer_type == 3:
        tokenizer = Tokenizer(WordPieceModel())
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordPieceTrainer(vocab_size=10000)
        tokenizer.train_from_iterator(sampled_documents, trainer=trainer)
        with open("output_wordpiece.dict", "w") as out_file:
            for word in tokenizer.get_vocab().keys():
                out_file.write(word + "\n")
            print("[DEBUG] WordPiece tokens written to output_wordpiece.dict.")
        tokens = tokenizer.encode(sampled_documents[0]).tokens
        print(tokens)

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) != 3:
        print("Usage: python script.py <dir> <tokenizer_type>")
        print("tokenizer_type: 0 for Simple, 1 for BPE, 2 for WordPiece")
        sys.exit(1)
    dir_path = sys.argv[1]
    tokenizer_type = int(sys.argv[2])
    main(dir_path, tokenizer_type)
    print(f"[DEBUG] Script executed in {time.time() - start_time:.2f} seconds.")
