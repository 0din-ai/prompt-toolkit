"""Tests for error types."""

from signature_sdk import (
    ConfigError,
    InvalidInputError,
    ModelError,
    ProviderError,
    SigError,
    parse_signature_string,
)


def test_sigerror_is_base():
    """Test SigError is base exception."""
    assert issubclass(ConfigError, SigError)
    assert issubclass(ProviderError, SigError)
    assert issubclass(ModelError, SigError)
    assert issubclass(InvalidInputError, SigError)


def test_sigerror_is_exception():
    """Test all error types inherit from Exception."""
    assert issubclass(SigError, Exception)
    assert issubclass(ConfigError, Exception)
    assert issubclass(ProviderError, Exception)
    assert issubclass(ModelError, Exception)
    assert issubclass(InvalidInputError, Exception)


def test_config_error():
    """Test ConfigError can be raised and caught."""
    try:
        raise ConfigError("Invalid configuration")
    except ConfigError as e:
        assert str(e) == "Invalid configuration"
    except Exception:
        assert False, "Should have caught ConfigError"


def test_provider_error():
    """Test ProviderError can be raised and caught."""
    try:
        raise ProviderError("API failure")
    except ProviderError as e:
        assert str(e) == "API failure"
    except Exception:
        assert False, "Should have caught ProviderError"


def test_model_error():
    """Test ModelError can be raised and caught."""
    try:
        raise ModelError("Model not found")
    except ModelError as e:
        assert str(e) == "Model not found"
    except Exception:
        assert False, "Should have caught ModelError"


def test_invalid_input_error():
    """Test InvalidInputError can be raised and caught."""
    try:
        raise InvalidInputError("Invalid input")
    except InvalidInputError as e:
        assert str(e) == "Invalid input"
    except Exception:
        assert False, "Should have caught InvalidInputError"


def test_catch_sigerror_catches_all():
    """Test catching SigError catches all subtypes."""
    for error_class in [ConfigError, ProviderError, ModelError, InvalidInputError]:
        try:
            raise error_class("test")
        except SigError:
            pass  # Expected
        except Exception:
            assert False, f"Should have caught {error_class.__name__} as SigError"


def test_parse_signature_string_raises_invalid_input_error():
    """Test parse_signature_string raises InvalidInputError on bad input."""
    try:
        parse_signature_string("invalid")
    except InvalidInputError as e:
        assert "Invalid signature prefix" in str(e)
    except Exception as e:
        assert False, f"Should raise InvalidInputError, got {type(e).__name__}"


def test_parse_signature_string_bad_version():
    """Test parse_signature_string raises InvalidInputError on bad version."""
    try:
        parse_signature_string("0din-v99:abcd1234")
    except InvalidInputError as e:
        assert "Unsupported signature version" in str(e)
    except Exception as e:
        assert False, f"Should raise InvalidInputError, got {type(e).__name__}"


def test_parse_signature_string_bad_hex():
    """Test parse_signature_string raises InvalidInputError on non-hex."""
    try:
        parse_signature_string("0din-v1:notahex!")
    except InvalidInputError as e:
        assert "Invalid hex signature" in str(e)
    except Exception as e:
        assert False, f"Should raise InvalidInputError, got {type(e).__name__}"
