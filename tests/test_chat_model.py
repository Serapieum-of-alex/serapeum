import os
import pytest
from serapeum.chat_model import ChatModel
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast

# Retrieve the Hugging Face token from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")


@pytest.fixture(scope="module")
def test_create_chat_model(model_id: str) -> ChatModel:
    # first test without tokenizer_kwargs
    ChatModel(model_id=model_id, access_token=huggingface_token, is_decoder=True)

    tokenizer_kwargs = {"clean_up_tokenization_spaces": True}

    chat_mode = ChatModel(
        model_id=model_id,
        access_token=huggingface_token,
        tokenizer_kwargs=tokenizer_kwargs,
        is_decoder=True,
    )
    assert chat_mode.model_id == model_id
    assert chat_mode.device == "cpu"
    assert isinstance(chat_mode.model, GemmaForCausalLM)
    assert isinstance(chat_mode.tokenizer, GemmaTokenizerFast)

    return chat_mode


class TestChatTemplate:
    def test_template_does_not_exist(self):
        template = ChatModel._read_chat_template(model_id="does-not-exist")
        assert template is None
