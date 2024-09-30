import logging
from uuid import uuid4

import polars as pl

from fasterstylometry.tokenization import tokenise_remove_pronouns_en


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


META_COLS = ['index', 'authors', 'books', 'texts', 'tokens']


class Corpus:
    def __init__(
        self,
        authors: list[str] = [],
        books: list[str] = [],
        texts: list[str] = [],
    ):
        self.df = pl.DataFrame(
            {
                'index': [str(uuid4()) for _ in range(len(authors))],
                'authors': authors,
                'books': books,
                'texts': texts,
            }
        )
        self._top_token_freqs = None
        self.z_scores = None


    def tokenise(
            self,
            tokeniser_expr: pl.Expr | callable = tokenise_remove_pronouns_en
        ):
        '''
        Tokenises the texts in the corpus and calculates the length of each tokenised text.
        The tokeniser_expr argument should be a polars expression that tokenises the texts,
        resulting in a list[str] column. The default tokeniser is a simple regex-based tokeniser
        that removes pronouns, apostrophes and tokens containing numbers.
        '''
        if isinstance(tokeniser_expr, pl.Expr):
            self.df = self.df.with_columns(
                tokeniser_expr
                .alias('tokens')
            )
        elif callable(tokeniser_expr):
            self.df = self.df.with_columns(
                pl.col('texts')
                .map_elements(
                    tokeniser_expr,
                    return_dtype=pl.List(pl.String)
                )
            )
        else:
            raise ValueError('tokeniser_expr must be a polars expression or a callable.')


    @property
    def top_token_freqs(
        self,
        k: int = 500,
        words_to_exclude: set = {},
        tok_match_pattern: str = r'.*'
    ) -> pl.DataFrame:
        '''Returns the top k most frequent tokens in the corpus. Only run explode if
        the column hasn't already been exploded.'''
        if self._top_token_freqs is None or k != self._top_token_freqs.width():
            self._top_token_freqs = self.df.select(
                pl.col('tokens')
                .explode()
                .value_counts(sort=True)
                .filter(
                    ~pl.col('value').is_in(words_to_exclude)
                    & pl.col('value').str.contains(tok_match_pattern)
                )
                .alias('token_counts')
            ).unnest('token_counts').limit(k)

        return self._top_token_freqs
    
    
    @property.setter(top_token_freqs)
    def top_token_freqs(self, value: pl.DataFrame):
        self._top_token_freqs = value
        

    def _calculate_token_counts_by_row(self):
        '''Returns the token counts for each book in the corpus.'''
        frequency_df = (
            self.df
            .explode('tokens')
            .filter(
                pl.col('tokens').is_in(self.top_token_freqs.select('tokens'))
            )
            .group_by(['index', 'tokens'])
            .agg(pl.len().alias('frequency'))
        )

        # TODO: Replace with group_by to allow lazy evaluation
        pivoted_df = (
            frequency_df.pivot(
                values='frequency',
                index='index',
                on='tokens',
                aggregate_function='first'
            )
            .fill_null(0)
        )

        self.df =  (
            self.df
            .select(META_COLS)
            .unique()
            .join(pivoted_df, on='index', how='left')
        )


    def _calculate_z_scores(self):
        columns_to_normalize = [
            col for col in self.df.columns if col not in META_COLS
        ]
            
        # Calculate mean and standard deviation once
        stats = self.df.select([
            pl.col(columns_to_normalize).mean().alias('mean'),
            pl.col(columns_to_normalize).std().alias('std')
        ])

        # Perform normalization
        normalized = self.df.select([
            pl.col('index'),
            pl.col('authors'),
            pl.col('books'),
            pl.col('texts'),
            pl.col('tokens'),
            *[((pl.col(col) - pl.col('mean')) / pl.col('std')).fill_nan(0).alias(col)
                for col in columns_to_normalize]
        ])

        return normalized.join(stats, how='cross')
    
    @property
    def z_scores(
        self,
        vocab_size: int = 500,
        words_to_exclude: set = {},
        tok_match_pattern: str = r'^[a-z][a-z]+$',
        force_recalculate: bool = False
    ) -> pl.DataFrame:
        '''Returns the z-scores for the the top k tokens in the corpus.
        This method will trigger the lazy evaluation of all the dataframes.
        
        Args:
            vocab_size: The number of top tokens to consider.
            words_to_exclude: A set of words to exclude from the top K tokens.
            tok_match_pattern: A regex pattern to match top K tokens.
            force_recalculate: If True, the cached stats will be ignored and recalculated.
        '''
        if self.z_scores is not None and not force_recalculate:
            return self.z_scores

        if 'tokens' not in self.df.columns or force_recalculate:
            logger.info('Corpus was not manually tokenised. Now tokenising with the default tokeniser.')
            # Clear cached stats
            self.top_token_freqs = None
            self.z_scores = None

            self.tokenise()

        self.top_token_freqs(
            k=vocab_size,
            words_to_exclude=words_to_exclude,
            tok_match_pattern=tok_match_pattern
        )

        self._calculate_token_counts_by_row()
        self._calculate_z_scores()

        # Return everything except the text and the tokens
        return self.df.select(
            pl.all().exclude(['texts', 'tokens'])
        )


    def calculate_burrows_delta(
            self,
            test_corpus: 'Corpus',
            vocab_size: int = 500,
            words_to_exclude: set = {},
            tok_match_pattern: str = r'^[a-z][a-z]+$'
        ) -> pl.DataFrame:
    
        # TODO: Turn calculation into a pipeline
        

        # Calculate self z-scores
        self._calculate_token_counts_by_row()

        if 'tokens' not in test_corpus.df.columns:
            logger.info('Test corpus was not manually tokenised. Now tokenising with the default tokeniser.')
            test_corpus.tokenise()

        # Set top_k tokens
        test_corpus.top_token_freqs = \
            self.top_token_freqs(
                k=vocab_size,
                words_to_exclude=words_to_exclude,
                tok_match_pattern=tok_match_pattern
            )
        
        # Calculate test z-scores
        test_corpus._calculate_token_counts_by_row()
        test_corpus._calculate_z_scores()

        # Calculate Burrows' Delta
        burrows_delta = (
            self.df
            .join(test_corpus.df, how='cross')
            .with_columns([
                (pl.col(f'df1.{col}') - pl.col(f'df2.{col}')).alias(col)
                for col in [col for col in self.df.columns if col not in ['index', 'authors', 'books', 'texts', 'tokens']]
            ])
        )

