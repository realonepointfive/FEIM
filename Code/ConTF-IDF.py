import json
import csv
import gzip
import os
import re
import ast
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import joblib
from joblib import Parallel, delayed
from scipy.sparse import vstack as sparse_vstack
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # progress bar optional


def load_msg_corpus(data_root, batch_dir=None):
    """Load message rows from Msg_batches if present, otherwise from Msg.txt.

    Args:
        data_root: Path to the dataset folder containing Msg.txt and Distinct_SE.txt.
        batch_dir: Optional explicit path to a directory containing Msg batch files.
    """
    if batch_dir is not None:
        if not os.path.isdir(batch_dir):
            raise FileNotFoundError(f"Specified Msg batch directory does not exist: {batch_dir}")
        target_dir = batch_dir
    else:
        target_dir = os.path.join(data_root, "Msg_batches")

    if os.path.isdir(target_dir):
        batch_files = sorted(
            [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.txt')]
        )
        if not batch_files:
            raise FileNotFoundError(f"Msg_batches directory exists but contains no .txt files: {target_dir}")

        print(f"Loading messages from {len(batch_files)} batch files in {target_dir}")
        corpus = []
        for batch_file in batch_files:
            with open(batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        corpus.append(line)
        return corpus

    msg_file = os.path.join(data_root, "Msg.txt")
    if not os.path.isfile(msg_file):
        raise FileNotFoundError(f"No Msg_batches directory or Msg.txt found under {data_root}")

    print(f"Loading messages from single file: {msg_file}")
    with open(msg_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def filter_conceptnet_csv(input_path, output_path):
    """
    Filters the raw assertions.csv for English concepts and creates a lookup map.
    
    Args:
        input_path: Path to 'assertions.csv' (or 'assertions.csv.gz')
        output_path: Path to save the filtered .json file
    """
    concept_map = defaultdict(set)
    count = 0
    
    # ConceptNet CSV format: uri, relation, start, end, metadata
    # We want to extract relations where both start and end are English (/c/en/)
    
    print("Filtering ConceptNet... this will take a few minutes.")
    
    # Use gzip if the file is compressed
    open_func = gzip.open if input_path.endswith('.gz') else open
    mode = 'rt' if input_path.endswith('.gz') else 'r'

    with open_func(input_path, mode, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        
        for row in reader:
            if len(row) < 4: continue
            
            start_node = row[2]
            end_node = row[3]
            
            # Check if both nodes are English concepts
            if start_node.startswith('/c/en/') and end_node.startswith('/c/en/'):
                # Strip prefix: /c/en/apple/n -> apple
                start_label = start_node.split('/')[3].lower()
                end_label = end_node.split('/')[3].lower()
                
                if start_label != end_label:
                    concept_map[start_label].add(end_label)
                    concept_map[end_label].add(start_label)
            
            count += 1
            if count % 1000000 == 0:
                print(f"Processed {count//1000000}M rows...")

    # Convert sets to lists for JSON serialization
    final_map = {k: list(v) for k, v in concept_map.items()}
    
    print(f"Saving filtered map with {len(final_map)} unique keywords...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_map, f)
    print("Done!")


class Con4GramModel:
    def __init__(self, local_json_path, svd_components=50):
        # 1. Load the local ConceptNet map we created earlier
        with open(local_json_path, 'r', encoding='utf-8') as f:
            self.concept_map = json.load(f)
        
        # 2. Configure TF-IDF for 4-grams of concepts
        # As per the text, we use 4-grams to keep clustering quality
        self.vectorizer = TfidfVectorizer(ngram_range=(4, 4), lowercase=True)
        
        # 3. SVD to map the high-dimensional vectors to 50 dimensions
        self.svd = TruncatedSVD(n_components=svd_components)

    def _extract_msg_id(self, message: str) -> str:
        """
        Extract the message ID from a Msg.txt line.
        Assumes the line contains a <msgid>...</msgid> tag.
        Falls back to empty string if not found.
        """
        m = re.search(r"<msgid>\s*(.*?)\s*</msgid>", message, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_timestamp_ms(self, message: str) -> str:
        """
        Extract timestamp in milliseconds from a Msg.txt line.
        Assumes the line contains a <timestamp_ms>...</timestamp_ms> tag.
        Falls back to empty string if not found.
        """
        m = re.search(r"<timestamp_ms>\s*(.*?)\s*</timestamp_ms>", message, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_userid(self, message: str) -> str:
        m = re.search(r"<userid>\s*(.*?)\s*</userid>", message, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _extract_rtuserid(self, message: str) -> str:
        m = re.search(r"<rtuserid>\s*(.*?)\s*</rtuserid>", message, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _extract_text_content(self, message: str) -> str:
        """
        Extract the logical text content of a Msg.txt line by combining
        <hashtags> and <text> (same behavior as the preprocessing script).
        """
        text_match = re.search(r"<text>\s*(.*?)\s*</text>", message, re.IGNORECASE)
        hashtags_match = re.search(r"<hashtags>\s*(.*?)\s*</hashtags>", message, re.IGNORECASE)

        text_content_parts = []
        if hashtags_match:
            hashtags = hashtags_match.group(1).strip()
            if hashtags:
                # Split multiple hashtags by whitespace
                text_content_parts.extend(hashtags.split())
        if text_match:
            text = text_match.group(1).strip()
            if text:
                text_content_parts.append(text)

        if text_content_parts:
            return " ".join(text_content_parts)
        else:
            # Fallback: treat the whole string as plain text
            return message

    def _get_concept_stream(self, message):
        """
        Converts a message M into a sequential stream of concepts {c_i^j}.
        Each keyword is replaced by its analogous concepts.
        """
        raw_text = self._extract_text_content(message)

        # Lowercase and split into tokens
        tokens = raw_text.lower().split()

        # Optionally avoid repetitive words in a single message
        # by keeping only the first occurrence of each token
        words = []
        seen = set()
        for w in tokens:
            if w not in seen:
                seen.add(w)
                words.append(w)
        concept_stream = []
        
        for w in words:
            # For each keyword w_i, find its concepts {c_i^j}
            concepts = self.concept_map.get(w, [w])
            # We append all concepts of this word to the stream
            # to allow n-grams to form across concept boundaries.
            concept_stream.extend(concepts)
            
        return " ".join(concept_stream)

    def fit(self, corpus):
        """
        Trains the model on a collection of messages.
        """
        # Convert all messages in corpus to concept streams
        concept_docs = [self._get_concept_stream(msg) for msg in corpus]
        
        # Construct ConTF/IDF vectors over the 4-grams
        tfidf_matrix = self.vectorizer.fit_transform(concept_docs)
        
        # Apply SVD to reduce dimensionality to 50
        self.svd.fit(tfidf_matrix)
        print(f"Model trained. Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def transform(self, message):
        """
        Creates the 50-dimensional topic vector for a message M.
        """
        # 1. Expand keyword sequence to concept sequence
        concept_doc = self._get_concept_stream(message)
        
        # 2. Extract 4-gram TF-IDF weights
        # This creates the vector V_i over the concept space
        v_sparse = self.vectorizer.transform([concept_doc])
        
        # 3. Map to lower 50-dimensional space using SVD
        v_reduced = self.svd.transform(v_sparse)
        
        return v_reduced[0]

    def transform_batch(self, corpus, show_progress=True, n_jobs=1):
        """
        Transforms many messages at once (much faster than calling transform in a loop).
        Returns 2D array of shape (len(corpus), svd_components).
        Set show_progress=True to display progress bars for concept extraction and
        vectorizer.transform (chunked mode).
        Set n_jobs>1 (or n_jobs=-1 for all cores) to parallelize vectorizer.transform.
        """
        iterator = tqdm(corpus, desc="concept_stream", unit="msg") if (show_progress and tqdm) else corpus
        concept_docs = [self._get_concept_stream(msg) for msg in iterator]

        # vectorizer.transform is single-threaded, so we chunk to enable both
        # parallel execution and user-visible progress.
        if len(concept_docs) > 0:
            if n_jobs == 1:
                # Keep chunking in single-thread mode so we can show transform progress.
                target_chunks = max(1, min(32, len(concept_docs)))
            else:
                n_jobs_actual = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                target_chunks = max(1, n_jobs_actual * 4)

            chunk_size = max(1, (len(concept_docs) + target_chunks - 1) // target_chunks)
            chunks = [concept_docs[i:i + chunk_size] for i in range(0, len(concept_docs), chunk_size)]

            chunk_iter = tqdm(chunks, desc="vectorizer.transform", unit="chunk") if (show_progress and tqdm) else chunks

            if n_jobs == 1:
                sparse_chunks = [self.vectorizer.transform(c) for c in chunk_iter]
            else:
                n_jobs_actual = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
                sparse_chunks = Parallel(n_jobs=n_jobs_actual)(
                    delayed(self.vectorizer.transform)(c) for c in chunk_iter
                )
            v_sparse = sparse_vstack(sparse_chunks)
        else:
            v_sparse = self.vectorizer.transform(concept_docs)

        # SVD transform is one matrix multiply (often BLAS-bound; use multi-threaded numpy if available)
        return self.svd.transform(v_sparse)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConTF-IDF model and generate message/sub-event concept vectors.")
    parser.add_argument("--data", type=str, default="WC2014",
                        help="Dataset subdirectory name under ../Data.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = os.path.join("..", "Data", args.data)
    concept_json_path = os.path.join("..", "Data", "conceptnet.json")
    se_path = os.path.join(data_root, "Distinct_SE.txt")
    model_output_path = os.path.join(data_root, "con4gram_model.pkl")
    msg_vectors_output_path = os.path.join(data_root, "Msg_concept_vectors.txt")
    se_vectors_output_path = os.path.join(data_root, "SE_concept_vectors.txt")

    msg_corpus = load_msg_corpus(data_root)

    if os.path.exists(model_output_path):
        model = joblib.load(model_output_path)
        print(f"Loaded existing model from {model_output_path}")
    else:
        model = Con4GramModel(concept_json_path, svd_components=50)
        model.fit(msg_corpus)
        joblib.dump(model, model_output_path)
        print(f"Trained and saved model to {model_output_path}")

    # 1) Msg_concept_vectors.txt format:
    #    (userid, rtuserid)\tspace-separated-vector\ttimestamp
    msg_vectors = model.transform_batch(msg_corpus, n_jobs=-1)
    with open(msg_vectors_output_path, "w", encoding="utf-8") as out_f:
        for idx, msg in enumerate(msg_corpus):
            uid = model._extract_userid(msg)
            ruid = model._extract_rtuserid(msg)
            edge = f"({uid}, {ruid})"
            timestamp_ms = model._extract_timestamp_ms(msg)
            vec_str = " ".join(str(float(x)) for x in msg_vectors[idx])
            out_f.write(f"{edge}\t{vec_str}\t{timestamp_ms}\n")
    print(f"Saved message concept vectors to {msg_vectors_output_path}")

    # 2) SE_concept_vectors.txt format:
    #    [v1, v2, ...]\ttimestamp
    se_texts = []
    se_times = []
    if not os.path.exists(se_path):
        raise FileNotFoundError(f"Distinct SE file not found: {se_path}")
    with open(se_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Keep the full tagged line so SE transformation is identical to Msg:
            # _get_concept_stream -> _extract_text_content (hashtags + text).
            se_texts.append(line)
            timestamp_ms = model._extract_timestamp_ms(line)
            se_times.append(timestamp_ms)

    se_vectors = model.transform_batch(se_texts, n_jobs=-1)
    with open(se_vectors_output_path, "w", encoding="utf-8") as out_f:
        for i in range(len(se_texts)):
            vec_str = "[" + ", ".join(str(float(x)) for x in se_vectors[i]) + "]"
            out_f.write(f"{vec_str}\t{se_times[i]}\n")
    print(f"Saved sub-event concept vectors to {se_vectors_output_path}")
