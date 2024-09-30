import logging

import polars as pl

from fasterstylometry import Corpus
from fasterstylometry.utils import META_COLS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BurrowsDelta:
    '''
    Measures the mean absolute distance between the z-scores of terms in a
    source and reference text. In this class, the delta metric is calculated
    between every pair of texts in the source and reference corpora.
    
    Args:
        train_corpus (Corpus): The source corpus.
        test_corpus (Corpus): The reference corpus.'''
    def __init__(
        self,
        train_corpus: Corpus,
        test_corpus: Corpus = None,
        vocab_size: int = 500,
        words_to_exclude: set = {},
        tok_match_pattern: str = r'^[a-z][a-z]+$'
    ):
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.vocab_size = vocab_size
        self.words_to_exclude = words_to_exclude
        self.tok_match_pattern = tok_match_pattern
        
        
    def calculate_burrows_delta(
        self,
        test_corpus: Corpus = None,
        vocab_size: int = None,
        words_to_exclude: set = None,
        tok_match_pattern: str = None,
        force_recalculate: bool = False
    ) -> pl.DataFrame:

        if test_corpus is not None and self.test_corpus is not None:
            logger.warning('New test corpus provided. Overwriting existing '
                           'test corpus.')
            self.test_corpus = test_corpus
        elif test_corpus is None and self.test_corpus is None:
            raise ValueError('A test corpus must be provided to calculate the '
                             'Burrows\' Delta.')

        # Calculate train z-scores 
        train_z_scores = self.train_corpus.z_scores(
            vocab_size=vocab_size or self.vocab_size,
            words_to_exclude=words_to_exclude or self.words_to_exclude,
            tok_match_pattern=tok_match_pattern or self.tok_match_pattern,
            force_recalculate=force_recalculate
        )

        # Set top_k tokens
        test_corpus.top_token_freqs = self.train_corpus.top_token_freqs

        # Calculate test z-scores
        test_z_scores = test_corpus.z_scores(
            vocab_size=vocab_size or self.vocab_size,
            words_to_exclude=words_to_exclude or self.words_to_exclude,
            tok_match_pattern=tok_match_pattern or self.tok_match_pattern,
            force_recalculate=force_recalculate
        )

        # Calculate Burrows' Delta
        burrows_delta = (
            self.df
            .join(test_corpus.df, how='cross')
            .with_columns([
                (pl.col(f'df1.{col}') - pl.col(f'df2.{col}')).alias(col)
                for col in [col for col in self.df.columns if col not in META_COLS]
            ])
        )