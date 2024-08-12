import os
import pytest
from serapeum.chat_model import ChatModel
from transformers.models.bert.modeling_bert import BertLMHeadModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast

# Retrieve the Hugging Face token from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token is None:
    raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")


@pytest.fixture(scope="module")
def test_create_chat_model() -> ChatModel:
    # first test without tokenizer_kwargs
    ChatModel(
        model_id="bert-base-uncased", access_token=huggingface_token, is_decoder=True
    )

    tokenizer_kwargs = {"clean_up_tokenization_spaces": True}

    chat_mode = ChatModel(
        model_id="bert-base-uncased",
        access_token=huggingface_token,
        tokenizer_kwargs=tokenizer_kwargs,
        is_decoder=True,
    )
    assert chat_mode.model_id == "bert-base-uncased"
    assert chat_mode.device == "cpu"
    assert isinstance(chat_mode.model, BertLMHeadModel)
    assert isinstance(chat_mode.tokenizer, BertTokenizerFast)

    return chat_mode


class TestChatTemplate:
    def test_template_does_not_exist(self):
        template = ChatModel._read_chat_template(model_id="does-not-exist")
        assert template is None
