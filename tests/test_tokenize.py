from semsearch.tokenize import tokenize


def test_tokenize_lowercases_and_keeps_digits() -> None:
    assert tokenize("SSO Reset 15-minute lock") == ["sso", "reset", "15", "minute", "lock"]


def test_tokenize_drops_punctuation_and_stopwords() -> None:
    assert tokenize("don't email a password!") == ["don", "t", "email", "password"]


def test_tokenize_can_keep_stopwords() -> None:
    assert tokenize("how do I reset", drop_stopwords=False) == ["how", "do", "i", "reset"]


def test_tokenize_empty() -> None:
    assert tokenize("   ...   ") == []
    assert tokenize("how do I") == []
