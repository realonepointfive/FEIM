import json
import csv
import gzip
import os
import re
import numpy as np
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


if __name__ == "__main__":
    """
    Train the Con4GramModel on NepalEQuake Msg.txt and:
      1) Save the trained model to disk
      2) Transform each message into a concept vector and save to a new txt file
    """
    # Paths relative to this script (Code directory)
    concept_json_path = r"..\Data\conceptnet.json"
    msg_path = r"..\Data\NepalEQuake\Distinct_SE.txt"
    model_output_path = r"..\Data\NepalEQuake\con4gram_model.pkl"
    vectors_output_path = r"..\Data\NepalEQuake\SE_concept_vectors.txt"

    # Load raw messages from Msg.txt (one line per message)
    with open(msg_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    # Load saved model if it exists; otherwise train and save
    if os.path.exists(model_output_path):
        model = joblib.load(model_output_path)
        print(f"Loaded existing model from {model_output_path}")
    else:
        model = Con4GramModel(concept_json_path, svd_components=50)
        model.fit(corpus)
        joblib.dump(model, model_output_path)
        print(f"Trained and saved model to {model_output_path}")

    # Transform all messages in one batch (much faster than per-message transform)
    # Format per line: [v1, v2, ... vK]<TAB>timestamp_ms, matching SE_Tokens.txt layout.
    vectors = model.transform_batch(corpus, n_jobs=-1)  # shape (n_messages, 50); -1 = all cores
    with open(vectors_output_path, "w", encoding="utf-8") as out_f:
        for idx, msg in enumerate(corpus):
            timestamp_ms = model._extract_timestamp_ms(msg)
            if not timestamp_ms:
                # Fallback to msgid/index to keep output stable if timestamp is missing.
                timestamp_ms = model._extract_msg_id(msg) or str(idx)
            vec_str = "[" + ", ".join(str(x) for x in vectors[idx]) + "]"
            out_f.write(f"{vec_str}\t{timestamp_ms}\n")

    print(f"Saved concept vectors to {vectors_output_path}")
