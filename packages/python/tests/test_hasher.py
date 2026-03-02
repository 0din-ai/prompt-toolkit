"""Tests for hasher abstraction."""

from odin_sig import Hasher, HashAlgorithm, LshConfig, SimHashLsh, get_hasher
from odin_sig.lsh import normalize_vector


def test_get_hasher_lsh():
    """Test get_hasher returns SimHashLsh for LSH algorithm."""
    hasher = get_hasher(HashAlgorithm.LSH)
    assert isinstance(hasher, SimHashLsh)
    assert hasher.name() == "lsh"


def test_get_hasher_unknown():
    """Test get_hasher raises ValueError for unknown algorithm."""
    try:
        get_hasher(HashAlgorithm.OPENAI)  # type: ignore
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)


def test_simhash_lsh_name():
    """Test SimHashLsh.name() returns 'lsh'."""
    hasher = SimHashLsh()
    assert hasher.name() == "lsh"


def test_simhash_lsh_compute():
    """Test SimHashLsh.compute() produces correct signatures."""
    hasher = SimHashLsh()
    vector = normalize_vector([1.0, 2.0, 3.0, 4.0])
    config = LshConfig(families=2, bits=64, bands=4)

    families = hasher.compute(vector, config)

    assert len(families) == 2
    assert families[0].family == 0
    assert families[1].family == 1
    assert families[0].bits == 64
    assert families[1].bits == 64
    assert len(families[0].signature) == 16  # 64 bits = 16 hex chars
    assert len(families[1].signature) == 16
    assert len(families[0].bands) == 4
    assert len(families[1].bands) == 4


def test_hasher_protocol():
    """Test that SimHashLsh conforms to Hasher protocol."""
    hasher: Hasher = SimHashLsh()
    vector = normalize_vector([0.5, 0.5, 0.5, 0.5])
    config = LshConfig(families=1, bits=256, bands=16)

    # Should satisfy the Hasher protocol
    assert callable(hasher.name)
    assert callable(hasher.compute)

    name = hasher.name()
    assert isinstance(name, str)

    families = hasher.compute(vector, config)
    assert isinstance(families, list)
    assert len(families) == 1
