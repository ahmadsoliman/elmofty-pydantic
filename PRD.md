# IslamQA AI - Product Requirements Document

## Project Overview

IslamQA AI is an intelligent question-answering system that provides accurate and sourced Islamic answers using state-of-the-art AI technology. The system processes user questions in multiple languages, leverages a comprehensive database of Islamic questions and answers, and delivers responses with appropriate sources and references.

## Product Vision

To provide the Muslim community worldwide with an accessible, accurate, and well-sourced AI assistant for Islamic inquiries that respects the depth and nuance of Islamic scholarship.

## Target Audience

- Muslims worldwide seeking religious guidance
- Islamic scholars and students
- Individuals interested in learning about Islam
- Mobile app users
- Telegram users

## Business Objectives

- Provide a reliable AI-powered Islamic Q&A service
- Make Islamic knowledge more accessible across language barriers
- Reduce misinformation by sourcing answers from established Islamic authorities
- Offer multiple access points (API, mobile app, Telegram bot)

## Core Features

### 1. Multi-language Processing

- Accept user questions in multiple languages
- Translation to Arabic for improved matching with source material
- Return answers in the user's original language

### 2. Retrieval-Augmented Generation (RAG)

- Vector database with embeddings of established Islamic Q&A content
- Semantic search to find the most relevant answers
- LLM-powered response generation based on retrieved content
- Source attribution to original Islamic reference materials

### 3. Multiple Interfaces

- RESTful API for developers and applications
- Streamlit web UI for direct interaction
- Telegram bot integration
- Mobile app integration

### 4. Security & Authentication

- Play Integrity API integration for Android app authentication
- Telegram verification for bot security
- Nonce-based request validation
- Rate limiting to prevent abuse

### 5. Caching & Performance

- Redis caching for improved response times
- Cloud SQL database for vector search
- Structured logging for monitoring and debugging

## Technical Architecture

### System Components

#### API Layer (Flask)

- Endpoints:
  - `/api/chat`: Main Q&A endpoint
  - `/api/telegram`: Telegram bot webhook
  - `/api/report`: User feedback/report submission
  - `/api/nonce`: Security token generation
  - `/api/health`: Service health checking
  - `/api/static`: Static file serving

#### Agent Pipeline

1. **Translation Agent**

   - Determines the language of user input
   - Translates non-Arabic questions to Arabic
   - Rewrites questions in multiple ways to improve matching

2. **Embedding Agent**

   - Generates vector embeddings for queries
   - Performs vector similarity search against the database
   - Returns the most relevant Q&A pairs as context

3. **Response Agent**
   - Generates a comprehensive answer using retrieved context
   - Formats the response in the user's language
   - Includes source references to original materials

#### Database

- PostgreSQL with vector search capabilities
- Tables:
  - `qas`: Stores question-answer pairs with embeddings
  - Vector indices for semantic search

#### Caching

- Redis for caching frequent queries
- TTL-based expiration policies

### Technology Stack

#### Backend

- Python 3.x
- Flask web framework
- Pydantic for data validation
- Structlog for structured logging
- Cohere for embeddings
- OpenRouter for LLM inference
- PostgreSQL for vector database
- Redis for caching

#### AI Components

- Pydantic AI framework for agent orchestration
- Cohere's multilingual embedding model
- OpenRouter for LLM inference
- Custom prompt templates for specialized agents

#### Deployment

- Google Cloud Platform
  - Cloud SQL for PostgreSQL
  - GCP service account integration
- Heroku compatibility (Procfile)
- Docker container support

## User Experience

### User Flows

#### Web UI Flow

1. User visits Streamlit interface
2. User enters a question about Islamic topics
3. System displays a loading indicator
4. System returns an answer with sources
5. User can ask follow-up questions

#### Telegram Bot Flow

1. User starts a conversation with the bot
2. Bot provides welcome message
3. User asks a question
4. Bot indicates it's processing
5. Bot responds with answer and sources

#### Mobile App Flow

1. User authenticates via Play Integrity
2. User submits a question
3. App displays a loading indicator
4. App presents the response with sources
5. User can report issues or inaccuracies

### User Experience Guidelines

- Responses must include source attribution when available
- Error messages should be informative but concise
- Arabic text should be properly displayed with correct RTL support
- Loading states should be indicated for operations over 1 second

## Technical Requirements

### Performance

- Response time: < 5 seconds for typical queries
- Availability: 99.9% uptime
- Concurrent requests: Support for at least 50 simultaneous users
- Caching: Frequently asked questions cached for 1 hour

### Security

- Input validation on all endpoints
- Play Integrity validation for Android app requests
- Telegram webhook validation
- Rate limiting to prevent abuse
- Sanitization of all user inputs
- Error handling that doesn't expose sensitive information

### Localization

- Support for English and Arabic queries and responses
- RTL layout support for Arabic in Streamlit UI
- Language detection for automatic processing

### Infrastructure

- Google Cloud SQL instance for vector database
- Redis for caching
- Logging integrated with monitoring systems
- Environment-based configuration for development/testing/production

## Implementation Details

### API Endpoints

#### `/api/chat` (POST)

- Accepts:
  ```json
  {
    "message": "string",
    "first_name": "string",
    "last_name": "string",
    "user_id": "string",
    "message_id": "string",
    "chat_id": "string",
    "chat_history": [
      {
        "id": "string",
        "text": "string",
        "sender": "user|bot",
        "timestamp": "datetime"
      }
    ],
    "integrity_token": "string"
  }
  ```
- Returns:
  ```json
  {
    "response": "string",
    "source_questions_ids": ["string"],
    "message": "string",
    "telegram_message": "string"
  }
  ```

#### `/api/telegram` (POST)

- Webhook endpoint for Telegram Bot API
- Validates Telegram-specific security headers
- Processes bot commands and queries

#### `/api/report` (POST)

- Accepts user feedback and issue reports
- Logs concerns for model improvement

### Database Schema

#### QA Table

- `id`: Primary key, question identifier
- `question`: The original question text in Arabic
- `answer`: The corresponding answer in Arabic
- `embedding`: Vector representation of the question

### Vector Search Implementation

- PostgreSQL with vector extension
- Functionality to match documents based on cosine similarity
- Configurable threshold for relevance matching
- Returns question IDs and similarity scores

## Testing Strategy

### Unit Testing

- Test all agent components individually
- Validate request/response schemas
- Test error handling

### Integration Testing

- Test the full query-response pipeline
- Verify translation accuracy
- Test vector search effectiveness

### Performance Testing

- Measure response times under various loads
- Test concurrent request handling
- Verify caching effectiveness

### User Acceptance Testing

- Test with real user queries
- Verify source attribution accuracy
- Assess response quality vs. ground truth

## Deployment Plan

### Environments

- Development: Local environment for developers
- Testing: Isolated environment for QA and testing
- Production: Live environment for end users

### Deployment Process

1. CI/CD pipeline to run tests
2. Containerization with Docker
3. Deployment to cloud provider
4. Database migrations for schema changes
5. Monitoring and logging setup

## Monitoring & Maintenance

### Key Metrics

- Response time by endpoint
- Error rates
- Query volume
- Cache hit rates
- Vector search effectiveness (similarity scores)

### Logging Strategy

- Structured logging with context
- Error tracking with stack traces
- Request/response logging (sanitized)
- Performance metrics

### Maintenance Tasks

- Regular database backups
- Model retraining as needed
- Log rotation and archiving
- Security patches and updates
- Redis cache maintenance

## Future Roadmap

### Short-term Enhancements

- Improved Arabic language model
- More comprehensive source attribution
- Enhanced error handling
- Expanded caching strategy

### Long-term Vision

- Additional language support
- Voice interface integration
- Image recognition for relevant Islamic texts
- Integration with more Islamic scholarly sources
- Community contribution features for scholars
