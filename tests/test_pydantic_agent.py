import pytest
from api.agents.embedding import get_embedding
from api.agents.orchesterator import (
    process_user_input,
)
from api.agents.response_agent import (
    ValidatedResponse,
    pydantic_islam_agent,
    run_response_agent,
)
from api.agents.translation_agent import (
    TranslationValidatedResponse,
    translation_agent,
)
from unittest.mock import patch, MagicMock


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_redis():
    with patch("redis.Redis") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@patch(
    "api.agents.embedding.cohere.ClientV2.embed",
    return_value=MagicMock(embeddings=MagicMock(float=[[0.1] * 1024])),
)
def test_get_embedding_success(mock_embed):
    # Test successful embedding generation

    result = get_embedding(["test query"])[0]
    assert len(result) == 1024
    assert all(x == 0.1 for x in result)


@patch("api.agents.embedding.cohere.ClientV2.embed", side_effect=Exception("API error"))
def test_get_embedding_failure(mock_embed):
    # Test embedding generation failure

    result = get_embedding(["test query"])
    assert len(result) == 0


@pytest.mark.anyio
async def test_run_agent_success():
    # Test successful agent run
    with patch.object(translation_agent, "run") as mock_translation_run:
        mock_translation_run.return_value = MagicMock(
            data=TranslationValidatedResponse(
                rewritten=["test query 1", "test query 2", "test query 3"],
                isArabic=False,
                language="english",
            )
        )

        with patch.object(pydantic_islam_agent, "run") as mock_response_run:
            mock_response_run.return_value = MagicMock(
                data=ValidatedResponse(
                    response="test response", source_questions_ids=["1", "2"]
                )
            )

            with patch("api.agents.embedding.generate_context") as mock_context:
                mock_context.return_value = "test context"

                result = await process_user_input("test query")
                assert "response" in result
                assert "source_questions_ids" in result
                assert "message" in result
                assert "telegram_mesasge" in result


@pytest.mark.anyio
async def test_run_agent_validation_error():
    # Test validation error when tool not used
    with patch.object(pydantic_islam_agent, "run") as mock_run:
        mock_run.side_effect = ValueError("Validation error")
        with pytest.raises(ValueError):
            await run_response_agent("test query", "", "english", [])


def test_validated_response_model():
    # Test ValidatedResponse model
    data = {"response": "test response", "source_questions_ids": ["1", "2"]}
    response = ValidatedResponse(**data)
    assert response.response == "test response"
    assert response.source_questions_ids == ["1", "2"]
