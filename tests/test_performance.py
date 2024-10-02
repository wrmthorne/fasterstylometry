import argparse
from glob import glob
import os
from time import perf_counter
import zipfile

import pandas as pd
import polars as pl

from fasterstylometry.tokenization import tokenise_remove_pronouns_en


def load_texts_with_polars(zip_file_path):
    """
    Load text files from a zip archive using Polars, storing them in a LazyFrame for efficient processing.
    """
    data = []
    with zipfile.ZipFile(zip_file_path, 'r') as archive:
        # Iterate through all text files in the archive
        for file_info in archive.infolist():
            if file_info.filename.endswith('.txt'):
                with archive.open(file_info.filename) as file:
                    text = file.read().decode('utf-8')
                    # Append filename and text content to the list
                    data.append((file_info.filename.replace('.txt', ''), text))

    # Create a LazyFrame for efficient lazy processing
    lf = pl.LazyFrame(data, schema=['filename', 'content'])
    return lf


def main(args):
    # Load the data
    pl_df = pl.read_csv(args.metadata_csv)
    pd_df = pd.read_csv(args.metadata_csv)

    zip_files = glob(os.path.join(args.root_dir, '*.zip'), recursive=True)

    start_time = perf_counter()
    lf_polars = load_texts_with_polars(args.root_dir)
    # Trigger evaluation if necessary
    lf_polars.collect()
    print(f"Polars loaded {len(zip_files)} files in time: {perf_counter() - start_time:.2f} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_csv', type=str, help='Path to the gutenberg metadata CSV file (Downloadable here: https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv).')
    parser.add_argument('--root_dir', type=str, help='Path to the directory containing the zip archives of the text files.')
    args = parser.parse_args()
    main(args)