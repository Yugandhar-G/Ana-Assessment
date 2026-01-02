# Conversational Ana AI Restaurant Search

This is an enhanced conversational interface for Ana AI that provides a hybrid approach to restaurant search.

## Features

### Hybrid Flow
1. **Direct Query First**: User asks a question and gets immediate results
2. **Follow-up Questions**: Ana asks contextual follow-up questions to better understand user preferences
3. **Refined Search**: If user answers follow-ups, Ana performs a refined search with combined query

### Key Components

1. **ConversationalQueryParser** (`src/conversational_query_parser.py`)
   - New query parser that combines original queries with follow-up answers
   - Separate from the existing `QueryParser` (doesn't affect the old flow)
   - Uses LLM to intelligently merge information into a comprehensive search query

2. **QuestionGenerator**
   - Generates contextual follow-up questions using LLM
   - Questions are relevant to the original query
   - Can cover vibe, occasion, dietary needs, location, etc.

3. **Conversational Gradio App** (`conversational_gradio_app.py`)
   - Chat interface with state management
   - Handles hybrid flow with follow-up questions
   - Users can skip questions with "skip" or "skip all"

## Usage

### Prerequisites
1. FastAPI server must be running on port 8000
   ```bash
   ./start_api.sh
   ```

2. Environment variables must be set (GEMINI_API_KEY, etc.)

### Starting the Conversational App

```bash
./start_conversational_gradio.sh
```

The app will start on port 7861 (separate from the regular Gradio app on 7860).

### Example Flow

```
User: "What desserts are famous in Maui?"
Ana: [Shows search results]
Ana: "Are you celebrating a special occasion?"

User: "Anniversary"
Ana: "Do you prefer a casual or romantic atmosphere?"

User: "Romantic"
Ana: [Performs refined search with: "desserts famous in Maui" + "anniversary" + "romantic atmosphere"]
     [Shows refined results]
```

### Commands

- **"skip"** - Skip current follow-up question
- **"skip all"** - Skip all remaining follow-up questions
- Just answer normally - Your answers will improve the refined search

## Architecture

The conversational flow works as follows:

1. User asks a direct question
2. System performs initial search and shows results
3. LLM generates 2-3 contextual follow-up questions
4. User can answer or skip questions
5. If answers provided, `ConversationalQueryParser` combines original query + answers
6. System performs refined search with combined query
7. Shows refined results

## Differences from Regular Gradio App

- **Regular App** (`gradio_app.py`): Single query → single search → results
- **Conversational App**: Query → search → follow-ups → refined search → results

Both apps use the same FastAPI backend and query parser infrastructure. The conversational app adds an additional layer for follow-up questions and query refinement.

