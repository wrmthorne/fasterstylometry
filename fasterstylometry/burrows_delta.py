import logging

import polars as pl
import polars.selectors as cs

from .corpus import Corpus
from .utils import META_COLS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BurrowsDelta:
    '''
    Measures the mean absolute distance between the z-scores of terms in a
    source and reference text. In this class, the delta metric is calculated
    between every pair of texts in the source and reference corpora.
    
    Args:
        train_corpus (Corpus): The source corpus.
        test_corpus (Corpus): The reference corpus.
        vocab_size (int): The size of the vocabulary to use in the analysis.
        words_to_exclude (set): A set of words to exclude from the analysis.
        tok_match_pattern (str): A regular expression pattern to match tokens.
    '''
    def __init__(
        self,
        train_corpus: Corpus,
        test_corpus: Corpus,
        vocab_size: int = 500,
        words_to_exclude: set = {},
        tok_match_pattern: str = r'^[a-z][a-z]+$'
    ):
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus
        self.vocab_size = vocab_size
        self.words_to_exclude = words_to_exclude
        self.tok_match_pattern = tok_match_pattern

        self._document_deltas = None

    def _calculate_burrows_delta(self):
        '''Protected method to calculate the Burrows' Delta across the provided
        source and reference corpora.'''

        self.test_corpus.top_k_tokens = self.train_corpus.top_k_tokens

        common_tokens = list(
            set(self.train_corpus.z_scores.columns)
            .intersection(self.test_corpus.z_scores.columns)
            .difference({'index', 'authors', 'titles'})
        )
        
        result = (
            self.train_corpus.z_scores
            .select(['index', 'authors', 'titles', *common_tokens])
            .join(
                self.test_corpus.z_scores.select(['index', 'authors', 'titles', *common_tokens]),
                how='cross',
                suffix='_test'
            )
        )

        return (
            result
            .with_columns([
                (pl.col(token) - pl.col(f'{token}_test')).abs().alias(f'delta_{token}')
                for token in common_tokens
            ])
            .with_columns([
                pl.mean_horizontal(pl.col('^delta_.*$')).alias('burrows_delta')
            ])
            .select([
                'index', 'authors', 'titles', 'index_test', 'authors_test', 'titles_test',
                'burrows_delta'
            ])
            .sort('burrows_delta')
        )


    @property
    def document_deltas(self) -> pl.DataFrame:
        '''
        Returns a DataFrame with the burrows deltas between every pair of
        documents in the source and reference corpora.'''
        if self._document_deltas is None:
            self._document_deltas = self._calculate_burrows_delta()

        return self._document_deltas


        
    