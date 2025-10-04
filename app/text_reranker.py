"""test"""
from itertools import product
from typing import List, Optional

import torch
from torch import Tensor
import numpy as np
import numpy.typing as npt
from transformers import AutoTokenizer, AutoModelForCausalLM

import starlette
from ray import serve

@serve.deployment
class Qwen3TextReranker:
    """A class for reranking text using the Qwen3 model."""
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen3-Reranker-0.6B',
            max_length: int = 8192,
            prefix: str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n",
            suffix: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.max_length = max_length

    @staticmethod
    def _format_instruction(
        query: str,
        doc: str,
        instruction: str = 'Given a search query, retrieve relevant passages that answer the query',
    ) -> str:
        output = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        return output

    def _process_inputs(self, pairs) -> Tensor:
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    @torch.no_grad()
    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
    async def rerank(
        self,
        query: str, # limit to 1 query
        documents: List[str],
        instruction: Optional[str] = None,
        **kwargs
    ) -> npt.NDArray[np.float16]:
        query_doc_permutation = []
        pairs = []
        query = [query]

        if instruction:
            instruction = [instruction]
            print(instruction)
            query_doc_permutation = list(product(query, documents, instruction))
            pairs = [self._format_instruction(query, doc, instruc) for query, doc, instruc in query_doc_permutation]
            print(pairs)
        else:
            query_doc_permutation = list(product(query, documents))
            pairs = [self._format_instruction(query, doc) for query, doc in query_doc_permutation]

        inputs = self._process_inputs(pairs)

        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().detach().cpu().numpy()
        return scores
    
    async def __call__(self, req: starlette.requests.Request):
        req = await req.json()
        return self.rerank(
            req["query"],
            req["documents"],
            req["instruction"],
        )
    
app = Qwen3TextReranker.bind()
