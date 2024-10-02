from uuid import UUID, uuid4

import numpy as np
import pytest
import polars as pl

from fasterstylometry.corpus import LazyCorpus, CorpusConfig  # Replace 'your_module' with the actual module name

@pytest.fixture
def sample_corpus():
    return LazyCorpus(
        authors=['Author1', 'Author2', 'Author3'],
        titles=['Title1', 'Title2', 'Title3'],
        texts=['This is text one', 'This is text two', 'This is text three']
    )

@pytest.fixture
def single_doc_corpus():
    return LazyCorpus(
        authors=['Author'],
        titles=['Title'],
        texts=['Single document text']
    )

def test_lazy_corpus_initialization(sample_corpus):
    assert isinstance(sample_corpus, LazyCorpus)
    assert isinstance(sample_corpus.lf, pl.LazyFrame)
    collected = sample_corpus.lf.collect()
    assert len(collected) == 3
    assert all(isinstance(UUID(index), UUID) for index in collected['index'])

def test_lazy_corpus_uneven_inputs():
    with pytest.raises(pl.exceptions.ShapeError):
        LazyCorpus(
            authors=['Author1', 'Author2'],
            titles=['Title1', 'Title2', 'Title3'],
            texts=['This is text one', 'This is text two']
        ).lf.collect()

def test_lazy_corpus_empty_inputs():
    corpus = LazyCorpus()
    collected = corpus.lf.collect()
    assert len(collected) == 0

def test_tokenise_with_expression(single_doc_corpus):
    config = CorpusConfig(tokeniser_expr=pl.col('texts').str.split(' '))
    corpus = LazyCorpus(
        authors=['Author1'],
        titles=['Title1'],
        texts=['This is a test'],
        config=config
    )
    corpus._tokenise()
    tokens = corpus.lf.collect()['tokens'].to_list()[0]
    assert 'tokens' in corpus.lf.columns
    assert tokens == ['This', 'is', 'a', 'test']

def test_tokenise_with_callable():
    def custom_tokenizer(text):
        return text.split()
    
    config = CorpusConfig(tokeniser_expr=custom_tokenizer)
    corpus = LazyCorpus(
        authors=['Author1'],
        titles=['Title1'],
        texts=['This is a test'],
        config=config
    )
    corpus._tokenise()
    tokens = corpus.lf.collect()['tokens'].to_list()[0]
    assert 'tokens' in corpus.lf.columns
    assert tokens == ['This', 'is', 'a', 'test']

def test_tokenise_invalid_expression():
    config = CorpusConfig(tokeniser_expr="invalid")
    corpus = LazyCorpus(
        authors=['Author1'],
        titles=['Title1'],
        texts=['This is a test'],
        config=config
    )
    with pytest.raises(ValueError, match='tokeniser_expr must be a polars expression or a callable.'):
        corpus._tokenise()

# ... rest of the test suite ...

def test_words_to_exclude():
    config = CorpusConfig(words_to_exclude=['is'])
    corpus = LazyCorpus(
        authors=['Author1', 'Author2'],
        titles=['Title1', 'Title2'],
        texts=['This is a test', 'This is another test'],
        config=config
    )
    token_stats = corpus._calculate_token_stats()
    stats = token_stats.collect()
    assert 'is' not in stats['tokens'].to_list()

def test_tok_match_pattern():
    config = CorpusConfig(tok_match_pattern=r'^[A-Za-z]+$')
    corpus = LazyCorpus(
        authors=['Author1', 'Author2'],
        titles=['Title1', 'Title2'],
        texts=['This is a test!', 'This is another test 123'],
        config=config
    )
    token_stats = corpus._calculate_token_stats()
    stats = token_stats.collect()
    assert all(token.isalpha() for token in stats['tokens'])

# Additional tests for edge cases
def test_empty_corpus():
    empty_corpus = LazyCorpus()
    assert len(empty_corpus.lf.collect()) == 0

def test_single_document_corpus():
    single_doc_corpus = LazyCorpus(
        authors=['Author'],
        titles=['Title'],
        texts=['Single document text']
    )
    assert len(single_doc_corpus.lf.collect()) == 1

def test_large_corpus():
    # Create a large corpus and test performance
    large_texts = ['Large text ' * 100] * 1000
    large_corpus = LazyCorpus(
        authors=['Author'] * 1000,
        titles=['Title'] * 1000,
        texts=large_texts)
    token_stats = large_corpus._calculate_token_stats()
    assert isinstance(token_stats, pl.LazyFrame)

def test_lazy_corpus_z_scores():
    config = CorpusConfig(
        tokeniser_expr=pl.col('texts').str.split(' '),
        tok_match_pattern=r'.*')
    corpus = LazyCorpus(
        authors=['Author1', 'Author2', 'Author3'],
        titles=['Title1', 'Title2', 'Title3'],
        texts=['a a b c', 'a b b c d e', 'd d d d e'],
        config=config
    )
    z_scores = corpus._calculate_z_scores().collect()
    assert not z_scores.is_empty()
    
    # Additional assertions to verify z-score calculation
    assert set(z_scores.columns) == {'index', 'tokens', 'z_score'}
    assert len(z_scores) > 0
    
    # Check if z-scores are calculated for each token
    unique_tokens = set(token for text in corpus.lf.collect()['texts'] for token in text.split())
    assert set(z_scores['tokens']) == unique_tokens

    expected_z_scores = np.array([
        [1.22474487, -0.57735027, -0.57735027, np.nan, np.nan],
        [-0.81649658,  0.57735027, -0.57735027, -1., -0.],
        [np.nan, np.nan, np.nan, 1., -0.]
    ])

    # Convert calculated z-scores to a similar structure as expected_z_scores
    tokens = ['a', 'b', 'c', 'd', 'e']
    calc_z_scores_array = np.full((3, 5), np.nan)
    
    for i, index in enumerate(z_scores['index'].unique()):
        for j, token in enumerate(tokens):
            mask = (z_scores['index'] == index) & (z_scores['tokens'] == token)
            if mask.sum() > 0:
                calc_z_scores_array[i, j] = z_scores.filter(mask)['z_score'][0]
    
    # Compare expected and calculated z-scores
    np.testing.assert_allclose(expected_z_scores, calc_z_scores_array, rtol=1e-5, atol=1e-8, equal_nan=True)

    # Verify that z-scores are numbers (not null or infinite)
    # z_scores = (
    #     corpus.lf
    #     .select(
    #         pl.all().exclude('tokens', 'texts')
    #     ).join(z_scores.lazy(), on='index', how='left')
    #     .select('authors', 'titles', 'tokens', 'z_score')
    #     .collect()
    #     .to_dict()
    # )

    # print(z_scores)

    # assert z_scores.select('authors', 'titles', 'tokens', 'z_score')