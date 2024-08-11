from typing import Union
import os
import torch
from torch import tensor
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import HfApi
from serapeum import __path__


ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
DEFAULT_MODELS_DIR = rf"{__path__[0]}/models"
if not os.path.exists(DEFAULT_MODELS_DIR):
    os.makedirs(DEFAULT_MODELS_DIR)
chat_templates_rdir = rf"{__path__[0]}/chat-templates"


class ChatModel:
    """
    ChatModel class to interact with a pre-trained language model for chatbot functionality.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2b-it",
        device="cpu",
        access_token: str = None,
        model_dir: str = None,
    ):
        """ChatModel class to interact with a pre-trained language model

        Parameters
        ----------
        model_id: str, default is "google/gemma-2b-it"
            model name as published in hugging face. possible models are:
            - "bert-base-uncased"
            - gpt2
            - roberta-base
            - distilbert-base-uncased
            - xlnet-base-cased
            - albert-base-v2
            - t5-base
            - meta-llama/Meta-Llama-3-8B-Instruct
            - meta-llama/Meta-Llama-3-8B
            - meta-llama/Llama-2-7b-hf
            - openai-community/openai-gpt
            - openai-community/gpt2-large
            - openai-community/gpt-3-5-turbo
            - openai-community/gpt-4
            - facebook/rag-token-base
        device: str, default is "cpu"
            cpu or cuda
        access_token: str, default is None.
            access token to hugging face.
        """
        if model_dir is None:
            model_dir = DEFAULT_MODELS_DIR

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory '{model_dir}' does not exist.")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=model_dir, token=access_token
        )

        self._model_id = model_id

        # To reduce memory
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            quantization_config = None

        self._device = device

        # TODO: try this parameter instead of the quantization_config `torch_dtype = torch.bfloat16`
        # create the tokenizer instance
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            quantization_config=quantization_config,
            cache_dir=model_dir,
            token=access_token,
        )
        # evaluation mode.
        self._model.eval()
        self.chat = []

    @property
    def model_id(self) -> str:
        """Hugging Face model name"""
        return self._model_id

    @property
    def device(self) -> str:
        """device"""
        return self._device

    @property
    def model(self) -> str:
        """chat model"""
        return self._model

    @property
    def tokenizer(self):
        """tokenizer"""
        return self._tokenizer

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        """generate.

            The generate function can be used to simply answer a question or to answer a question with additional context
            (which we will retrieve from documents).

            Generate a response to a question.

        Parameters
        ----------
        question: str
            The question to answer.
        context: str, optional
            The context for the question. Defaults to None.
        max_new_tokens: int, optional
            The maximum number of tokens to generate. Defaults to 250.
        """
        if context is None or context == "":
            prompt = f"""Give a detailed answer to the following question. Question: {question}"""
        else:
            prompt = f"""
            Using the information contained in the context, give a detailed answer to the question.
            Context: {context}.
            Question: {question}
            """

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable assistant who helps users with their questions.",
            },
            {"role": "user", "content": f"{prompt}"},
            {"role": "assistant", "content": "You are a helpful assistant."},
        ]
        # get the chat template.
        self.tokenizer.chat_template = self.get_chat_template()
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            # do not add tokens to the chat message (start and end strings) "<s>[INST] Hello, how are you? [/INST]
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids: tensor = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs: tensor = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # remove input prompt from response
        response = response[len(formatted_prompt) :]
        # remove eos token
        response = response.replace("<eos>", "")

        return response

    def get_chat_template(self) -> str:
        """Get chat template.

        Warnings
        -------
        If the chat template file does not exist, a warning will be raised and the default template will be used.

        Returns
        -------
        chat_template: str
            the jinja2 chat template.
        """
        template = (
            self._read_chat_template(model_id=self.model_id)
            if None
            else (self.tokenizer.default_chat_template)
        )
        return template

    @staticmethod
    def _read_chat_template(model_id: str) -> Union[str, None]:
        """Read chat template

        Parameters
        ----------
        model_id: str
            the model id as the hugging face model name.

        Warnings
        -------
        If the chat template file does not exist, a warning will be raised and the default template will be used.

        Returns
        -------
        chat_template: str
            the jinja2 chat template.
        """
        chat_template_file = f"{model_id.replace('/', '!')}.jinja"
        path = f"{chat_templates_rdir}/{chat_template_file}"
        if os.path.exists(path):
            chat_template = open(path).read()
            chat_template = chat_template.replace("    ", "").replace("\n", "")
        else:
            warnings.warn(
                f"The mode: {model_id} does not have a template, therefore the default template will be used"
            )
            chat_template = None
        return chat_template


def list_models(limit: int = 10, author: str = None, detailed: bool = False):
    """List models from hugging face

    Parameters
    ----------
    limit: int, default is 10
        limit number of models
    author: str
        author of the models (i.e., google, openai)
    detailed: bool, optional, default False
        If True, return detailed information about each model.

    Returns
    -------
    list_of_models: list
    """
    api = HfApi()
    models = list(api.list_models(limit=limit, author=author))

    # if not detailed, get only the id
    if not detailed:
        models = [item.id for item in models]

    return models
