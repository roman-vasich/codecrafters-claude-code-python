def test_open_ai_tool_schema_describes_required_field():
    from pydantic import BaseModel

    from app.models import OpenAIProvider
    # Arrange: define a small Pydantic request model.

    # Act: call OpenAIProvider.tool_schema with that model.

    # Assert: check the relevant parts of the returned dictionary.
    class Model(BaseModel):
        arg1: str
        arg2: int
        arg3: float
        arg4: bool

    schema = OpenAIProvider.tool_schema(Model, name="model", description="desc")

    assert schema is not None
    assert "function" in schema, f"Expected {schema} to contain function"
    assert "parameters" in schema["function"], f"Expected {
        schema
    } to contain parameters"
    assert len(schema["function"]["parameters"]["properties"].keys()) == 4


def test_anthropic_tool_schema_describes_required_field():
    from pydantic import BaseModel

    from app.models import AnthropicProvider
    # Arrange: define a small Pydantic request model.

    # Act: call OpenAIProvider.tool_schema with that model.

    # Assert: check the relevant parts of the returned dictionary.
    class Model(BaseModel):
        arg1: str
        arg2: int
        arg3: float
        arg4: bool

    schema = AnthropicProvider.tool_schema(Model, name="model", description="desc")

    assert schema is not None
    assert "input_schema" in schema, f"Expected {
        schema
    } to contain input_schema property"
    assert "type" in schema["input_schema"], f"Expected {schema} to contain type"
    assert "properties" in schema["input_schema"], f"Expected {schema} to contain parameters"
    assert len(schema["input_schema"]["properties"].keys()) == 4
