import logging

import polars as pl

from .corpus import Corpus


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
    def __init__(self, train_corpus: Corpus, test_corpus: Corpus):
        self.train_corpus = train_corpus
        self.test_corpus = test_corpus

    @property
    def document_deltas(self) -> pl.DataFrame:
        """
        Returns a DataFrame with the burrows deltas between every pair of
        documents in the source and reference corpora.
        """
        document_deltas = (
            # TODO: Find way to remove rows where there is no token overlap
            self.train_corpus.z_scores
            .join(self.test_corpus.z_scores, how='cross', suffix='_test')
            .filter(pl.col('z_score_test').is_not_null())
            .with_columns(
                (pl.col('z_score') - pl.col('z_score_test')).abs().alias('delta')
            )
            .group_by([
                'index', 'author', 'title',
                'index_test', 'author_test', 'title_test'
            ])
            .agg(pl.mean('delta').alias('burrows_delta'))
            .sort('burrows_delta', descending=False)
        )
        return document_deltas.collect()