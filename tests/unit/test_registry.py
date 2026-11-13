import pytest
from pydantic import BaseModel

from ragkit.core.registry import Registry, StageSpec


class GreeterParams(BaseModel):
    greeting: str = "hello"


def make_registry() -> Registry:
    reg = Registry("greeter")

    @reg.register("plain", GreeterParams)
    def _plain(params: GreeterParams, name: str = "world"):
        return f"{params.greeting} {name}"

    return reg


def test_create_validates_params_and_passes_dependencies():
    reg = make_registry()
    assert reg.create(StageSpec(impl="plain", greeting="hi"), name="ada") == "hi ada"
    assert reg.create({"impl": "plain"}) == "hello world"


def test_unknown_impl_lists_available_names():
    with pytest.raises(KeyError, match="plain"):
        make_registry().create({"impl": "missing"})


def test_unknown_param_is_rejected():
    with pytest.raises(ValueError, match="greting"):
        make_registry().create({"impl": "plain", "greting": "typo"})


def test_duplicate_registration_fails():
    reg = make_registry()
    with pytest.raises(ValueError):
        reg.register("plain")(lambda params: None)
