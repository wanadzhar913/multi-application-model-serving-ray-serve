"""test"""
from typing import List, Optional

import torch
from torch import Tensor
import numpy as np
import numpy.typing as npt
from transformers import AutoTokenizer, AutoModel

import starlette
from ray import serve

@serve.deployment
class Qwen3TextEmbedding:
    """A class for generating text embeddings using the Qwen3 model."""
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen3-Embedding-0.6B',
            max_length: int = 8192,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.max_length = max_length

    @staticmethod
    def _last_token_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Pools the last token representation, handling both left and right padding."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        """Formats a query with a given task description."""
        return f'Instruct: {task_description}\nQuery: {query}'

    @torch.no_grad()
    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
    async def embed(
            self,
            texts: List[str],
            instruction: Optional[str] = None,
        ) -> npt.NDArray[np.float16]:
        """
        Generates normalized embeddings for input text(s).

        Args:
            texts: A list of strings to embed.
            instruction: Optional task instruction for formatting queries.

        Returns:
            A list of embeddings (list of floats per text).
        """
        if instruction:
            texts = [self._get_detailed_instruct(instruction, text) for text in texts]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**batch_dict)
        embeddings = self._last_token_pool(
            outputs.last_hidden_state, batch_dict['attention_mask']
        ).detach().cpu().numpy()
        return embeddings
    
    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.embed(
            req["texts"],
            req["instruction"]
        )

app = Qwen3TextEmbedding.bind()
