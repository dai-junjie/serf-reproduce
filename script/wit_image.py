#!/usr/bin/env python3
"""
Convert WIT image ResNet embeddings from CSV.gz to fvecs format.

Fvecs format:
- For each vector: 4 bytes (int32 dimension) + dim * 4 bytes (float values)

Usage:
    python wit_image.py /path/to/resnet_embeddings/ output.fvecs --limit 1000000
"""

import gzip
import csv
import struct
import argparse
from pathlib import Path


def read_csv_gz_embeddings(file_path):
    """Read embeddings from a CSV.gz file.

    Yields:
        tuple: (image_url, embedding_list)
    """
    with gzip.open(file_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['image_url']
            embedding = [float(x) for x in row['embedding'].split(',')]
            yield url, embedding


def write_fvecs(embeddings, output_path, limit=None):
    """Write embeddings to fvecs format.

    Args:
        embeddings: Iterator of (url, embedding_list) tuples
        output_path: Output fvecs file path
        limit: Maximum number of vectors to write (None = all)
    """
    count = 0
    with open(output_path, 'wb') as f:
        for url, emb in embeddings:
            if limit is not None and count >= limit:
                break

            dim = len(emb)
            # Write dimension (int32)
            f.write(struct.pack('i', dim))
            # Write float values
            for val in emb:
                f.write(struct.pack('f', val))

            count += 1
            if count % 10000 == 0:
                print(f"Written {count} vectors...")

    print(f"Total written: {count} vectors to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description='Convert WIT ResNet embeddings to fvecs')
    parser.add_argument('input_dir', help='Directory containing test_resnet_embeddings_part-*.csv.gz files')
    parser.add_argument('output_file', help='Output fvecs file path')
    parser.add_argument('--limit', type=int, default=1000000,
                        help='Maximum number of vectors to extract (default: 1000000)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    # Find all embedding files
    embedding_files = sorted(input_dir.glob('test_resnet_embeddings_part-*.csv.gz'))

    if not embedding_files:
        print(f"Error: No embedding files found in {input_dir}")
        return 1

    print(f"Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"  - {f.name}")

    def embedding_generator():
        for file_path in embedding_files:
            print(f"Processing {file_path.name}...")
            yield from read_csv_gz_embeddings(file_path)

    write_fvecs(embedding_generator(), output_file, args.limit)
    return 0


if __name__ == '__main__':
    exit(main())
