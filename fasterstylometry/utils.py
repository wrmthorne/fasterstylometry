from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

from .tokenization import tokenise_remove_pronouns_en

META_COLS = ['index', 'authors', 'titles', 'texts', 'tokens']


@dataclass
class DeltaConfig:
    k: int = field(
        default=500,
        metadata={'help': 'The size of the vocabulary to use in the analysis (default: %(default)s).'}
    )
    words_to_exclude: set[str] = field(
        default_factory=set,
        metadata={'help': 'A set of words to discard during analysis.'}
    )
    tok_match_pattern: str = field(
        default=r'^[a-z][a-z]+$',
        metadata={'help': 'A regular expression pattern to match tokens (default: %(default)s).'}
    )
    tokeniser_expr: 'pl.Expr' | Callable[[str], list[str]] = field(
        default_factory=lambda: tokenise_remove_pronouns_en,
        metadata={'help': 'A function to tokenise the texts.'}
    )