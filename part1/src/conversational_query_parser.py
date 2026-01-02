"""Conversational Query Parser - Combines original queries with follow-up answers."""
import json
import os
from typing import Dict
from .gemini_client import AsyncGeminiClient


class ConversationalQueryParser:
    """Query parser that combines original query with follow-up answers into a comprehensive search query."""
    
    def __init__(self, client: AsyncGeminiClient | None = None):
        self.client = client or AsyncGeminiClient(
            api_key=os.getenv("GEMINI_API_KEY"),
            default_chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        )
    
    async def combine_query_and_follow_ups(
        self,
        original_query: str,
        follow_up_answers: Dict[str, str]
    ) -> str:
        """Combine original query with follow-up answers into a comprehensive query.
        
        Args:
            original_query: The user's initial question/query
            follow_up_answers: Dictionary mapping question keys to user's answers
            
        Returns:
            A combined, comprehensive search query that incorporates all information
        """
        
        if not follow_up_answers:
            return original_query
        
        system_prompt = """You are a query refinement system. Your job is to combine an original user query with follow-up answers into a single, comprehensive search query that captures all the user's preferences and context.

Guidelines:
- Preserve the original query intent and information
- Naturally integrate the follow-up answers into the query
- Create a query that sounds natural, not robotic
- Include all relevant details from both the original query and follow-ups
- Make it suitable for restaurant search

Example:
Original: "What desserts are famous in Maui?"
Follow-ups: {"vibe": "romantic", "occasion": "anniversary", "dietary": "no nuts"}
Combined: "Find romantic restaurants in Maui famous for desserts, perfect for anniversary, with no nuts in desserts"

Respond with ONLY the combined query, nothing else."""
        
        # Format follow-up answers
        follow_up_text = "\n".join([f"- {key}: {value}" for key, value in follow_up_answers.items()])
        
        user_prompt = f"""Original query: "{original_query}"

Follow-up answers:
{follow_up_text}

Combine these into a single comprehensive search query:"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.client.default_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=300,
            )
            
            combined_query = response.choices[0].message["content"].strip()
            # Remove quotes if LLM added them
            if combined_query.startswith('"') and combined_query.endswith('"'):
                combined_query = combined_query[1:-1]
            
            return combined_query
        except Exception as e:
            print(f"Error combining query: {e}")
            # Fallback: simple concatenation
            follow_up_text = ", ".join([f"{key}: {value}" for key, value in follow_up_answers.items()])
            return f"{original_query}. Additional preferences: {follow_up_text}"

