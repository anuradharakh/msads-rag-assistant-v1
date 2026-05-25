import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from src.generation.prompt_builder import build_grounded_prompt


class AnswerGenerator:
    """OPENAI-BASED GROUNDED ANSWER GENERATOR. **"""

    def __init__(
        self,
        llm_config: Dict,
        generation_config: Dict,
    ):
        load_dotenv()

        self.provider = llm_config.get("provider", "openai")
        self.model_name = llm_config["name"]
        self.temperature = llm_config.get("temperature", 0.0)
        self.max_tokens = llm_config.get("max_tokens", 700)
        self.fallback_message = generation_config["fallback_message"]

        if self.provider != "openai":
            raise ValueError("Only OpenAI provider is supported in this version.")

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        self.client = OpenAI(api_key=api_key)

    def generate_from_prompt(self, prompt: str) -> str:
        """GENERATE TEXT FROM PROMPT. **"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content

    def generate(
        self,
        question: str,
        chunks: List[Dict],
    ) -> str:
        """GENERATE GROUNDED ANSWER. **"""

        prompt = build_grounded_prompt(
            question=question,
            chunks=chunks,
            fallback_message=self.fallback_message,
        )

        return self.generate_from_prompt(prompt)