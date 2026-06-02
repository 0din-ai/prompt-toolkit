"""Tests for the high-level sus_factor() entry point."""

# sus_factor imports from compare.py which imports classifier.py which raises
# ImportError if torch is absent. Defer all imports that pull in the heavy
# chain to inside the test functions so pytest can collect this file without
# torch installed.
from odin_prompt_toolkit.susfactor.types import SusFactorResult


class FakeClassifier:
    """Implements the classify/close surface used by sus_factor()."""

    def __init__(self, score=0.9, label="suspicious"):
        self._score = score
        self._label = label
        self.closed = False

    async def classify(self, text):
        return SusFactorResult(
            score=self._score,
            label=self._label,
            model="fake",
            threshold=0.5,
            timing_ms=1.0,
        )

    async def close(self):
        self.closed = True


async def test_uses_provided_classifier():
    from odin_prompt_toolkit.susfactor import sus_factor

    clf = FakeClassifier(score=0.91, label="suspicious")
    result = await sus_factor("hack the prompt", classifier=clf)
    assert result.score == 0.91
    assert result.label == "suspicious"


async def test_does_not_close_caller_owned_classifier():
    """A classifier passed in by the caller must not be closed."""
    from odin_prompt_toolkit.susfactor import sus_factor

    clf = FakeClassifier()
    await sus_factor("text", classifier=clf)
    assert clf.closed is False


async def test_auto_constructs_and_closes(monkeypatch):
    """When no classifier is given, one is built and then closed."""
    from odin_prompt_toolkit.susfactor import classifier as classifier_mod
    from odin_prompt_toolkit.susfactor import sus_factor

    created = {}

    async def fake_new(cls, cache, **kwargs):
        clf = FakeClassifier(score=0.2, label="safe")
        created["clf"] = clf
        return clf

    monkeypatch.setattr(
        classifier_mod.SusFactorClassifier,
        "new",
        classmethod(fake_new),
    )

    result = await sus_factor("benign text")
    assert isinstance(result, SusFactorResult)
    assert result.label == "safe"
    assert created["clf"].closed is True
