from uuid import uuid4

import polars as pl

from .utils import DeltaConfig, META_COLS


class Corpus:
    def __init__(
            self,
            authors: list[str] = [],
            titles: list[str] = [],
            texts: list[str] = [],
            config: DeltaConfig = DeltaConfig
        ()
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


class LazyCorpus:
    def __init__(
        self,
        authors: list[str] = [],
        titles: list[str] = [],
        texts: list[str] = [],
        config: DeltaConfig = DeltaConfig
    ()
    ):
        self.lf = pl.LazyFrame({
            'index': [str(uuid4()) for _ in range(len(authors))],
            'authors': authors,
            'titles': titles,
            'texts': texts,
        })
        self.config = config
        self._tokenised = False
        self._top_k_tokens = None
        self._token_stats = None
        self._z_scores = None

    def _tokenise(self):
        if not self._tokenised:
            if isinstance(self.config.tokeniser_expr, pl.Expr):
                self.lf = self.lf.with_columns(self.config.tokeniser_expr.alias('tokens'))
            elif callable(self.config.tokeniser_expr):
                self.lf = self.lf.with_columns(
                    pl.col('texts').map_elements(
                        self.config.tokeniser_expr,
                        return_dtype=pl.List(pl.String)
                    ).alias('tokens')
                )
            else:
                raise ValueError('tokeniser_expr must be a polars expression or a callable.')
            
            # Always go through and remove excluded tokens and check pattern match
            self.lf = (
                self.lf.select([
                    pl.col('*'), pl.col('tokens')
                    .list.eval(pl.element().filter(
                        ~pl.element().is_in(self.config.words_to_exclude) &
                        pl.element().str.contains(self.config.tok_match_pattern)))
                ])
            )

            self._tokenised = True

    def _calculate_token_stats(self):
        self._tokenise()

        # Calculate token frequency per row
        row_token_freqs = (
            self.lf
            .explode('tokens')
            .group_by(['index', 'tokens'])
            .agg(pl.len().alias('frequency'))
        )

        # Calculate top K tokens across the corpus if one wasn't provided
        # in the case this is a test corpus
        if self._top_k_tokens is None:
            self._top_k_tokens = (
                row_token_freqs
                .group_by('tokens')
                .agg(pl.sum('frequency').alias('token_freqs'))
                .sort('token_freqs', descending=True)
                .limit(500)
                .select('tokens')
            )

        z_scores = (
            row_token_freqs
            .join(self._top_k_tokens, on='tokens', how='inner')
            .select([
                pl.col('*'),
                (pl.col('frequency') - pl.mean('frequency').over('tokens')
                    / pl.std('frequency').over('tokens')).alias('z_score')
            ])
        )

        self._z_scores = (
            self.lf
            .select('index', 'authors', 'titles')
            .join(z_scores, on='index', how='left')
        )

    def clear_cache(self) -> None:
        self._top_k_tokens = None
        self._z_scores = None
        self._tokenised = False
        self.lf = self.lf.drop('tokens', strict=False)

    @property
    def top_k_tokens(self) -> (pl.LazyFrame | pl.DataFrame): 
        if self._top_k_tokens is None:
            self._calculate_token_stats()
        return self._top_k_tokens

    @top_k_tokens.setter
    def top_k_tokens(self, value: (pl.LazyFrame | pl.DataFrame)):
        if isinstance(value, (pl.LazyFrame | pl.DataFrame)):
            self.clear_cache()
            self._top_k_tokens = value
        else:
            raise ValueError('top_k_tokens must be a either a DataFrame or '
                             'preferably a LazyFrame.')

    @property
    def z_scores(self) -> (pl.LazyFrame | pl.DataFrame):
        if self._z_scores is None:
            self._calculate_token_stats()
        return self._z_scores