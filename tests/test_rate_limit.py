import httpx
from openai import RateLimitError
import pytest
from unittest.mock import patch, MagicMock

from agent import call_with_retry


def create_mock_rate_limit_error():
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return RateLimitError(
        message="Rate limit reached",
        response=response,
        body={"error": {"message": "Rate limit reached"}},
    )


@patch("agent.time.sleep")
@patch("agent.client.chat.completions.create")
def test_call_with_retry_success_first_try(mock_create, mock_sleep):
    mock_response = MagicMock()
    mock_create.return_value = mock_response

    result = call_with_retry(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert result == mock_response
    assert mock_create.call_count == 1
    mock_sleep.assert_not_called()


@patch("agent.time.sleep")
@patch("agent.client.chat.completions.create")
def test_call_with_retry_recovers_after_rate_limit(mock_create, mock_sleep):
    mock_success = MagicMock()
    rate_error = create_mock_rate_limit_error()

    mock_create.side_effect = [rate_error, rate_error, mock_success]

    result = call_with_retry(messages=[], tools=[])

    assert result == mock_success
    assert mock_create.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


@patch("agent.time.sleep")
@patch("agent.client.chat.completions.create")
def test_call_with_retry_exceeds_max_attempts(mock_create, mock_sleep):
    rate_error = create_mock_rate_limit_error()
    mock_create.side_effect = rate_error

    with pytest.raises(RuntimeError, match="failed after all attempts"):
        call_with_retry(messages=[], tools=[], max_retries=3)

    assert mock_create.call_count == 3
    assert mock_sleep.call_count == 3
    assert [call.args[0] for call in mock_sleep.call_args_list] == [1, 2, 4]