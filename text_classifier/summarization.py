import logging


from concurrent.futures import as_completed, ThreadPoolExecutor
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from text_classifier.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_SUMMARY_WORDS
from transformers import GPT2TokenizerFast, pipeline 
from typing import List, Optional


class Summarizer:

    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.bart_pipe = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            truncation=True,
            num_beams=4,
            early_stopping=True,
            top_p=0.9,
        )
        self.chain_llm = HuggingFacePipeline(pipeline=self.bart_pipe)
        self.abs_pipe = self.bart_pipe

    def _make_token_splitter(
        self, chunk_size: int, chunk_overlap: int
    ) -> CharacterTextSplitter:
        return CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def summarize_map_reduce(self, text: str) -> str:
        txt = text.strip()
        if not txt:
            return ""

        splitter = self._make_token_splitter(CHUNK_SIZE, CHUNK_OVERLAP)
        chunks = splitter.split_text(txt)
        docs = [Document(page_content=chunk) for chunk in chunks if chunk]
        if not docs:
            return ""

        map_prompt = PromptTemplate(
            template="""
                Summarize:

                {text}
            """,
            input_variables=["text"],
        )
        combine_prompt = PromptTemplate(
            template="""
                Summarize:

                {text}
            """,
            input_variables=["text"],
        )

        chain = load_summarize_chain(
            self.chain_llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )
        try:
            return chain.run(docs).strip()
        except Exception as e:
            logging.error(f"Map-Reduce summarization failed: {e}")
            return ""

    def abstractive_summary(
        self,
        text: str
    ) -> str:
        max_workers=5
        txt = text.strip()
        if not txt:
            return ""

        tokens = self.tokenizer(txt)["input_ids"]
        if len(txt.split()) < 10:
            logging.warning("Text too short to summarize effectively.")
            return txt

        def summarize_chunk(chunk: str) -> str:
            out = self.abs_pipe(
                chunk,
                min_length=MIN_SUMMARY_WORDS,
                max_length=self.abs_pipe.tokenizer.model_max_length,
                do_sample=False,
                num_beams=4,
                early_stopping=True,
                top_p=0.9,
            )
            return out[0]["summary_text"].strip()

        if len(tokens) > CHUNK_SIZE:
            splitter = self._make_token_splitter(CHUNK_SIZE, 0)
            parts = splitter.split_text(txt)
            summaries: List[str] = []
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(summarize_chunk, p): p for p in parts}
                for fut in as_completed(futures):
                    try:
                        summaries.append(fut.result())
                    except Exception as e:
                        logging.error(f"Chunk summarization failed: {e}")
            return " ".join(summaries)

        return summarize_chunk(txt)

