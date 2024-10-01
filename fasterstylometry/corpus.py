from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

import polars as pl

from .tokenization import tokenise_remove_pronouns_en
from .utils import META_COLS



@dataclass
class CorpusConfig:
    k: int = field(
        default=500,
        metadata={'help': 'The size of the vocabulary to use in the analysis.'}
    )
    words_to_exclude: set[str] = field(
        default_factory=set,
        metadata={'help': 'A set of words to discard during analysis.'}
    )
    tok_match_pattern: str = field(
        default=r'^[a-z][a-z]+$',
        metadata={'help': 'A regular expression pattern to match tokens.'}
    )
    tokeniser_expr: pl.Expr | Callable[[str], list[str]] = field(
        default_factory=lambda: tokenise_remove_pronouns_en,
        metadata={'help': 'A function to tokenise the texts.'}
    )


class Corpus:
    def __init__(
            self,
            authors: list[str] = [],
            titles: list[str] = [],
            texts: list[str] = [],
            config: CorpusConfig = CorpusConfig()
    ):
        self.df = pl.DataFrame({
            'index': [str(uuid4()) for _ in range(len(authors))],
            'authors': authors,
            'titles': titles,
            'texts': texts,
        })
        self._unique_tokens = None
        self._top_k_tokens = None
        self._z_scores = None

        self.config = config

    def _tokenise(self):
        if isinstance(self.config.tokeniser_expr, pl.Expr):
            self.df = self.df.with_columns(self.config.tokeniser_expr.alias('tokens'))
        elif callable(self.config.tokeniser_expr):
            self.df = self.df.with_columns(
                pl.col('texts').map_elements(
                    self.config.tokeniser_expr,
                    return_dtype=pl.List(pl.String)
                ).alias('tokens')
            )
        else:
            raise ValueError('tokeniser_expr must be a polars expression or a callable.')
        
    def _calculate_token_frequency_stats(self):
        if 'tokens' not in self.df.columns:
            self._tokenise()

        token_counts = (
            self.df
            .explode('tokens')
            .filter(
                ~pl.col('tokens').is_in(self.config.words_to_exclude) &
                pl.col('tokens').str.contains(self.config.tok_match_pattern)
            )
            .group_by(['index', 'tokens'])
            .agg(pl.count().alias('frequency'))
        )

        self._top_k_tokens = (
            token_counts
            .group_by('tokens')
            .agg(pl.sum('frequency').alias('total_frequency'))
            .sort('total_frequency', descending=True)
            .limit(self.config.k)
            .select('tokens')
        )

        return (
            token_counts
            .join(self._top_k_tokens, on='tokens', how='inner')
            .pivot(
                values='frequency',
                index='index',
                on='tokens',
                aggregate_function='first'
            )
            .fill_null(0)
        )
    
    def _calculate_z_scores(self):             
        # TODO: Come back and optimise i.e. remove repeat mean and std calcs
        z_scores_df = (
            self._calculate_token_frequency_stats()
            .with_columns([
                ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
                .fill_null(0)
                .alias(col)
                for col in self.df.columns if not col in META_COLS
            ])
        )
        self._z_scores = (
            self.df.select('index', 'authors', 'titles')
            .join(z_scores_df, on='index', how='left')
        )
    
    @property
    def top_k_tokens(self) -> pl.DataFrame:
        # TODO: Reimplement so only top_k needs calculating?
        if self._top_k_tokens is None:
            self._calculate_token_frequency_stats()
        return self._top_k_tokens
    
    @top_k_tokens.setter
    def top_k_tokens(self, value: pl.DataFrame):
        # TODO: need to conditionally invalidate cached data if tokens are different
        self._top_k_tokens = value
    
    @property
    def z_scores(self) -> pl.DataFrame:
        if self._z_scores is None:
            self._calculate_z_scores()
        return self._z_scores
    
    @property
    def unique_tokens(self) -> set[str]:
        if self._unique_tokens is None:
            self._unique_tokens = set(self.z_scores.columns) - set(META_COLS)

        return self._unique_tokens