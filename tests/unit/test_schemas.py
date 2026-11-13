from ragkit.core.schemas import Chunk, GeneratedAnswer


def test_indexed_text_prefixes_breadcrumb():
    c = Chunk(
        chunk_id="a::c0",
        doc_id="a",
        text="Use Depends.",
        title="Dependencies",
        section="Dependencies > First steps",
    )
    assert c.indexed_text == "Dependencies > First steps\nUse Depends."


def test_indexed_text_without_section_is_plain_text():
    assert Chunk(chunk_id="a::c0", doc_id="a", text="x").indexed_text == "x"


def test_generated_answer_schema_is_flat_json():
    schema = GeneratedAnswer.model_json_schema()
    assert schema["properties"]["status"]["enum"] == ["answered", "insufficient_context"]
