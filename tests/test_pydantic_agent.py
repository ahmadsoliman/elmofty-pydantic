import pytest
from api.pydantic_agent import (
    get_embedding,
    run_agent,
    RAGToolTracker,
    ValidatedResponse,
    pydantic_islam_agent,
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
    "api.pydantic_agent.cohere.ClientV2.embed",
    return_value=MagicMock(embeddings=MagicMock(float=[[0.1] * 1024])),
)
def test_get_embedding_success(mock_embed):
    # Test successful embedding generation

    result = get_embedding("test query")
    assert len(result) == 1024
    assert all(x == 0.1 for x in result)


@patch("api.pydantic_agent.cohere.ClientV2.embed", side_effect=Exception("API error"))
def test_get_embedding_failure(mock_embed):
    # Test embedding generation failure

    result = get_embedding("test query")
    assert len(result) == 1024
    assert all(x == 0 for x in result)


@pytest.mark.anyio
async def test_run_agent_success():
    # Test successful agent run
    with patch.object(pydantic_islam_agent, "run") as mock_run:
        RAGToolTracker.set_used()
        mock_run.return_value = MagicMock(
            data=ValidatedResponse(
                response="test response", source_questions_ids=["1", "2"]
            )
        )

        result = await run_agent("test query")
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
            await run_agent("test query")


def test_rag_tool_tracker():
    # Test RAG tool tracker functionality
    RAGToolTracker.reset()
    assert RAGToolTracker.check() is False
    RAGToolTracker.set_used()
    assert RAGToolTracker.check() is True
    RAGToolTracker.reset()
    assert RAGToolTracker.check() is False


def test_validated_response_model():
    # Test ValidatedResponse model
    data = {"response": "test response", "source_questions_ids": ["1", "2"]}
    RAGToolTracker.set_used()
    response = ValidatedResponse(**data)
    assert response.response == "test response"
    assert response.source_questions_ids == ["1", "2"]
