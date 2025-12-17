import json
from pathlib import Path
from openai import AsyncOpenAI
from .schemas import ParsedQuery, MustNotFilters, Preferences, SignalWeights


class QueryParser:
    """Parse natural language queries into structured search intent using LLM."""
    
    def __init__(self, client: AsyncOpenAI | None = None):
        self.client = client or AsyncOpenAI()
        self.system_prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).parent.parent / "prompts" / "query_parser.txt"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "Parse the user's restaurant query into structured search intent."
    
    async def parse(self, query: str) -> ParsedQuery:
        """Parse a natural language query into structured form."""
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        parsed_json = json.loads(response.choices[0].message.content)
        
        return ParsedQuery(
            raw_query=query,
            semantic_query=parsed_json.get("semantic_query", query),
            must_not=MustNotFilters(**parsed_json.get("must_not", {})),
            preferences=Preferences(**parsed_json.get("preferences", {})),
            weights=SignalWeights(**parsed_json.get("weights", {})),
            location=parsed_json.get("location"),
        )

