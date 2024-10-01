from uuid import uuid4

import polars as pl

from fasterstylometry.tokenization import tokenise_remove_pronouns_en
from fasterstylometry.utils import META_COLS


class Corpus:
    def __init__(self, authors: list[str] = [], books: list[str] = [], texts: list[str] = []):
        self._ldf = pl.LazyFrame({
            'index': [str(uuid4()) for _ in range(len(authors))],
            'authors': authors,
            'books': books,
            'texts': texts,
        })
        self._top_token_freqs = None
        self._z_scores = None

        self._top_token_freqs = None
        self._z_scores = None
        self._k = 500  # Default value for k
        self._words_to_exclude = set()
        self._tok_match_pattern = r'.*'


    def tokenise(self, tokeniser_expr: pl.Expr | callable = tokenise_remove_pronouns_en):
        if isinstance(tokeniser_expr, pl.Expr):
            self._ldf = self._ldf.with_columns(tokeniser_expr.alias('tokens'))
        elif callable(tokeniser_expr):
            self._ldf = self._ldf.with_columns(
                pl.col('texts').map_elements(
                    tokeniser_expr,
                    return_dtype=pl.List(pl.String)
                ).alias('tokens')
            )
        else:
            raise ValueError('tokeniser_expr must be a polars expression or a callable.')
    

    def calculate_top_token_freqs(self, k: int = 500, words_to_exclude: set = set(), tok_match_pattern: str = r'.*'):
        self._k = k
        self._words_to_exclude = words_to_exclude
        self._tok_match_pattern = tok_match_pattern
        
        self._top_token_freqs = (
            self._ldf.select(pl.col('tokens'))
            .explode('tokens')
            .group_by('tokens')
            .agg(pl.count().alias('count'))
            .filter(
                ~pl.col('tokens').is_in(words_to_exclude)
                & pl.col('tokens').str.contains(tok_match_pattern)
            )
            .sort('count', descending=True)
            .limit(k)
        )


    def _calculate_token_counts_by_row(self):
        return (
            self._ldf.explode('tokens')
            .filter(pl.col('tokens').is_in(self.top_token_freqs.select('tokens')))
            .group_by(['index', 'tokens'])
            .agg(pl.count().alias('frequency'))
            .group_by('index')
            .agg([
                pl.col('tokens').alias('token'),
                pl.col('frequency')
            ])
            .with_columns([
                pl.col('token').list.to_struct().alias('token_freqs')
            ])
            .unnest('token_freqs')
            .with_columns([
                pl.col('^(?!index).*$').fill_null(0)
            ])
        )


    def _calculate_z_scores(self):
        token_counts = self._calculate_token_counts_by_row()
        columns_to_normalize = [col for col in token_counts.columns if col not in ['index']]
        
        stats = token_counts.select([
            pl.col(columns_to_normalize).mean().alias('mean'),
            pl.col(columns_to_normalize).std().alias('std')
        ])

        normalized = token_counts.select([
            pl.col('index'),
            *[((pl.col(col) - pl.col('mean')) / pl.col('std')).fill_nan(0).alias(col)
              for col in columns_to_normalize]
        ])

        return self._ldf.select(META_COLS).join(normalized, on='index', how='left').join(stats, how='cross')


    @property
    def top_token_freqs(self):
        if self._top_token_freqs is None:
            self.calculate_top_token_freqs(self._k, self._words_to_exclude, self._tok_match_pattern)
        return self._top_token_freqs
    

    @property.setter
    def top_token_freqs(self, value: pl.DataFrame):
        # TODO: Need to invalidate stored data if vocabularies don't match
        self._top_token_freqs = value

        # only invalidate if the vocabulary is different
        if self._z_scores_lazy is not None and isinstance(self._z_scores_lazy, pl.DataFrame):
            if self._z_scores_lazy.width() != value.width():
                self._z_scores_lazy = None


    @property
    def z_scores(self) -> pl.DataFrame | pl.LazyFrame:
        if self._z_scores_lazy is None:
            self._z_scores_lazy = self._calculate_z_scores().select
        return self._z_scores_lazy


    def get_z_scores(self, columns: list[str] = None) -> pl.DataFrame:
        """
        Fetch z-scores, optionally selecting specific columns.
        This method triggers computation and returns a DataFrame.
        """
        query = self.z_scores
        if columns:
            query = query.select(['index', 'authors', 'books'] + columns)
        return query.collect()
    
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame):
        return cls(
            authors=df['authors'].to_list(),
            books=df['books'].to_list(),
            texts=df['texts'].to_list()
        )