from uuid import uuid4

import polars as pl

from .tokenization import tokenise_remove_pronouns_en


class Corpus:
    def __init__(self, authors=[], titles=[], texts=[], top_k_tokens=None, tokenise_expr=tokenise_remove_pronouns_en):
        self._lf = pl.LazyFrame({
            'index': [str(uuid4()) for _ in range(len(authors))],
            'author': authors,
            'text': texts,
            'title': titles
        })

        # Setup in init but no executed until called collected
        self._lf = self._lf.with_columns(tokenise_expr.alias('tokens'))

        _row_token_freqs = (
            self._lf
            .explode('tokens')
            .group_by(['index', 'author', 'title', 'tokens'])
            .agg(pl.len().alias('frequency'))
        )

        self.top_k_tokens = top_k_tokens
        if top_k_tokens is None:
            self.top_k_tokens = (
                _row_token_freqs
                .group_by('tokens')
                .agg(pl.sum('frequency').alias('token_freqs'))
                .sort('token_freqs', descending=True)
                .limit(500)
                .select('tokens')
            )

        self.z_scores = (
            _row_token_freqs
            .join(self.top_k_tokens, on='tokens', how='inner')
            .select([
                pl.col('*'),
                (
                    (pl.col('frequency') - pl.mean('frequency').over('tokens')) /
                    (pl.std('frequency').over('tokens') + 1e-10) # Avoid division by zero
                )
                .fill_null(0) # STD of 1 element with ddof=1 is null
                .alias('z_score')
            ])
        )