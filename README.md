# FasterStylometry


### Tokenising

The `Corpus` class exposes a `tokenise` method. A default tokeniser will be used if none are supplied. The default tokeniser is a polars reimplementation of the `tokenise_remove_pronouns_en` from the original FastStylometry package which removes english pronouns, removes apostrophes and splits words using `\w+`.

To submit a callable as the tokenise function, your function must accept a `str` and return a `list[str]`. The text field for each row will be passed to the method one at a time until the full series has been processed.

```python
# Faststylometry implementation
def tokenise_remove_pronouns_en(text: str) -> list:
    text_normalised = re.sub("['’]", "", text.lower())
    tokens = [tok for tok in re_words.findall(text_normalised) if not is_number_pattern.match(tok)]

    tokens_without_pronouns = [tok for tok in tokens if tok not in PRONOUNS]

    return tokens_without_pronouns
```

`NOTE: **It is strongly recommended to use polars expressions rather than callables!**`

`map_elements` uesd in the callable tokenising strategy breaks polars' computation graph meaning it cannot be automaticallty optimised. The tokeniser is also applied row by row meaning the benefit of parallelism is also lost. Using Polars expressions retains all of these performance benefits. The expression should first state the columns to ingest - almost always `texts`. The expression must eventually resolve to `pl.List(pl.String)` which is the list of tokens for each text. Here is the same method implemented in polars.

```python
# Default tokeniser 
tokenise_remove_pronouns_en = (
    pl.col('texts')
    .str.to_lowercase()
    .str.replace_all("[`']", '')
    .str.extract_all('\w+')
    .list.eval(
        pl.element().filter(
            ~pl.element().is_in(PRONOUNS) | ~pl.element().str.contains('.*\d+.*')
        )
    )
)
```