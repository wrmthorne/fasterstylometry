# Faster Stylometry

<!-- badges: start -->

![my badge](https://badgen.net/badge/Status/In%20Development/orange)

[![PyPI package](https://img.shields.io/badge/pip%20install-fasterstylometry-brightgreen)](https://pypi.org/project/fasterstylometry/) [![version number](https://img.shields.io/pypi/v/fasterstylometry?color=green&label=version)](https://github.com/wrmthorne/fasterdatascience/releases) [![License](https://img.shields.io/github/license/wrmthorne/fasterstylometry)](https://github.com/wrmthorne/fasterstylometry/blob/main/LICENSE)

<!-- badges: end -->

The faster and more memory conscious implementation of [Fast Stylometry](https://github.com/fastdatascience/faststylometry/tree/main) Wood, T. A. (2024) for calculating the Burrows Delta Metric for stylistic similarity analysis and author attribution. The API is inspired by the original library but major refactoring was done to move responsibility for relevant calculations to the corpus. Additionally, everything has been rewritten to run in almost pure Polars.

The metric measures the distance between a text's token frequency distribution and a set of reference texts' token frequency distributions. Smaller deltas indicate a higher stylistic similarity.

The two key mathematical components are the `z-score` and `delta` value. A `z-score` is maintained for each token for each text and is calculated based on the frequency of a certain token $x$ appearing in the text, the mean frequency of that token across all texts in a corpus $\mu$, and the standard deviation of the frequency of the token across the corpus $\sigma$ - in short it is a normalisation function:

$$
z = \frac{x - \mu}{\sigma}
$$

The Delta score is the actual distance measure between a test and a reference text. In summary it is the mean absolute distance between the two text's z-scores: 

$$
\Delta = \frac{1}{n} \sum^{n}_{i=1} | z_i^{\text{target}} - z_i^{\text{reference}} |
$$

where $n$ is the number of words in the feature space, $z_i^{\text{target}}$ is z-score of word $i$ in the target text and $z_i^{\text{reference}}$ the same but for the reference text.

**This library is a bit of fun and is in no way meant to demean or offend the authors of Fast Stylometry**.
I have gotten good use out of the library but I needed something faster and more scalable for my current work.

## Installation

The library can be installed through PyPi. There are optionally `dev` and `all` builds to install the testing libraries and the libraries used for one off/small scale experimentation respectively.

```bash
# TOOD: COMING AT SOME POINT
pip install fasterstylometry # Optionally [dev] or [all]
```

## Corpus

### Loading a Corpus

The `Corpus` is the main entry point to the library. It holds the collection of authors, titles and texts and maintains all the logic for tokenising and calculating the relevant statistics. There are a number of ways to instantiate a `Corpus` object, the first and simplest is as any normal class would be instantiated. A `Corpus` may also be instantiated from an existing dataframe of the correct format.

```python
# Basic iinstantiatoin
corpus = Corpus(
    authors = ['Hermann Melville', 'Jane Austen', 'Philip K. Dick']
    books = ['Moby Dick', 'Pride and Prejudice', 'Do Androids Dream of Electric Sheep']
    texts = ['Call me Ishmael ...', 'It is a truth universally ...', 'A merry little surge of ...']
)

# From existing polars dataframe of correct format
corpus = Corpus.from_dataframe(df)
```

Finally, in the same way as Fast Stylometry, a `Corpus` may be loaded from a collection of text files in a directory. Directories may be nested or flat but the filenames must be of the format `[author]_-_[title].txt` where the contents of the file is only the book text.

```python
corpus = Corpus.from_dir('path/to/dir/')
```

### Using the Corpus

Most of the methods for the `Corpus` are intentionally protected as they are performed in a particular sequence to calculate z-values. A `Corpus` is intended to be passed to the Delta class to perform the stylometry. The main method of interest is `tokenise` which is described in greater detail below. Even after instantiating a  `Corpus` internal statistics won't be calculated as they are implemented as lazy operations in Polars. This allows the computational graph to be as optimised as possible before starting processing. Only once the `z_score` method is called will the majority of calculations be poerformed. These are then cached in the `Corpus` so it may be reused without repeating the expense.

```python
from fasterstylometry import BurrowsDelta, Corpus

# Instantiate train and test corpora
train_corpus = ...
test_corpus = ...

delta = BurrowsDelta(
    train_corpus = train_corpus,

    # Optional Parameters
    test_corpus = test_corpus,
    vocab_size = 500,
    words_to_exclude = {},
    tok_match_pattern: str = r'^[a-z][a-z]+$'
)

# TODO: Returns the full matrix with all delta permutations
diff_matrix = delta.calculate_burrows_delta()

# TODO: Add example

# TODO: Returns the delta values between each test document and an aggregate#
# z-score for each author in the train corpus
delta.author_deltas

# TODO: Add example
```

The library is setup to potentially accommodate other delta metrics in the future but this may never happen.

## Tokenising

The `Corpus` class exposes a `tokenise` method. A default tokeniser will be used if no updated tokeniser is provided. The default tokeniser is a polars reimplementation of the `tokenise_remove_pronouns_en` from the original FastStylometry package which removes english pronouns, removes apostrophes and splits words using `\w+`.

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


## Running Tests

To run the tests, make sure the `dev` dependencies are installed as described above. To run the tests, use the following command:

```bash
# LOL You think I have written any tests yet
```

## References

Wood, T. A. (2024). Fast Stylometry (Computer software) (Version 1.0.4) [Computer software]. https://doi.org/10.5281/zenodo.11096941


## TODO
- [ ] add parquet caching for large datasets that exceed memory
- [ ] Fix lazy evaluation of dataframes
- [ ] Unit tests
- [ ] Add 
- [ ] Finish README
- [ ] CI/CD
- [ ] Setup on PyPi