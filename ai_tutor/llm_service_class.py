import openai
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from django.db import connection
from django.core.exceptions import ValidationError
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class LLMService:
    """
    Enhanced LLM service with improved error handling, security, and functionality
    Optimized for Django and integrated with the existing RagPipeLine
    """
    
    def __init__(self):
        """Initialize the LLM service with proper configuration"""
        try:
            # Validate OpenAI API key
            if not hasattr(settings, 'OPENAI_API_KEY') or not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key not configured in settings")
            
            openai.api_key = settings.OPENAI_API_KEY
            
            # Initialize embedding model with error handling
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to initialize local embedding model: {e}")
                self.embedding_model = None
            
            # Configuration constants
            self.MAX_CONTEXT_LENGTH = 4000
            self.MAX_QUERY_LENGTH = 8000  # Increased for content upload
            self.DEFAULT_TEMPERATURE = 0.7
            self.DEFAULT_MAX_TOKENS = 500
            
            logger.info("LLMService initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMService: {str(e)}")
            raise

    def _validate_text_input(self, text: str, max_length: int = None) -> str:
        """Validate and sanitize text input"""
        if not isinstance(text, str):
            raise ValidationError("Input must be a string")
        
        text = text.strip()
        if not text:
            raise ValidationError("Input cannot be empty")
        
        if max_length and len(text) > max_length:
            # Truncate instead of raising error for content upload
            logger.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text content with retry logic and fallback
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            ValidationError: If input is invalid
            Exception: If both OpenAI and local embedding fail
        """
        try:
            # Validate and potentially truncate input
            text = self._validate_text_input(text, self.MAX_QUERY_LENGTH)
            
            # Try OpenAI first
            try:
                response = openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                
                embedding = response.data[0].embedding
                logger.debug(f"Generated OpenAI embedding of length {len(embedding)}")
                return embedding
                
            except Exception as openai_error:
                logger.warning(f"OpenAI embedding failed: {openai_error}")
                
                # Fallback to local model
                if self.embedding_model:
                    try:
                        embedding = self.embedding_model.encode(text)
                        logger.debug(f"Generated local embedding of length {len(embedding)}")
                        return embedding.tolist()
                    except Exception as local_error:
                        logger.error(f"Local embedding failed: {local_error}")
                        raise Exception(f"Both OpenAI and local embedding failed: {openai_error}, {local_error}")
                else:
                    raise Exception(f"OpenAI embedding failed and no local model available: {openai_error}")
                    
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def semantic_search(self, query: str, limit: int = 5, grade_filter: str = None, topic_filter: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the same logic as RagPipeLine
        This is a wrapper around the existing semantic search functionality
        
        Args:
            query: Search query
            limit: Maximum number of results
            grade_filter: Optional grade filter
            topic_filter: Optional topic filter
            
        Returns:
            List of similar content dictionaries
        """
        try:
            # Import here to avoid circular imports
            from .rag_pipeline import RagPipeLine
            
            # Use the existing RagPipeLine for semantic search
            rag_pipeline = RagPipeLine()
            results = rag_pipeline.semantic_search(
                query=query,
                top_k=limit,
                grade_filter=grade_filter,
                topic_filter=topic_filter
            )
            
            logger.info(f"Semantic search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []

    def _prepare_context(self, context: str) -> str:
        """Prepare and truncate context to fit model limits"""
        if len(context) > self.MAX_CONTEXT_LENGTH:
            # Truncate context intelligently (try to end at sentence boundary)
            truncated = context[:self.MAX_CONTEXT_LENGTH]
            last_sentence = truncated.rfind('.')
            if last_sentence > self.MAX_CONTEXT_LENGTH * 0.8:
                truncated = truncated[:last_sentence + 1]
            
            logger.warning(f"Context truncated from {len(context)} to {len(truncated)} characters")
            return truncated
        
        return context

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_answer(self, question: str, context: str, persona_prompt: str = "") -> str:
        """
        Generate answer using RAG with improved prompting and error handling
        
        Args:
            question: User's question
            context: Retrieved context
            persona_prompt: Optional persona/role prompt
            
        Returns:
            Generated answer
        """
        try:
            # Validate inputs
            question = self._validate_text_input(question, 500)
            context = self._validate_text_input(context)
            
            # Prepare context
            context = self._prepare_context(context)
            
            # Build system prompt
            base_prompt = """You are an educational tutor. Use the following context to answer the student's question.

Instructions:
- Base your answer primarily on the provided context
- If the context doesn't contain relevant information, politely say so and provide general guidance
- Be clear, concise, and educational
- Use examples when helpful
- Maintain a supportive and encouraging tone

Context:
{context}"""
            
            if persona_prompt:
                system_prompt = f"{persona_prompt}\n\n{base_prompt.format(context=context)}"
            else:
                system_prompt = base_prompt.format(context=context)
            
            # Generate response
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=self.DEFAULT_MAX_TOKENS,
                temperature=self.DEFAULT_TEMPERATURE,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated answer")
            
            return answer
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise Exception(f"Failed to generate answer: {str(e)}")

    def _get_database_schema(self) -> str:
        """Get current database schema information for content tables"""
        try:
            with connection.cursor() as cursor:
                # Get table information for content-related tables
                cursor.execute("""
                    SELECT table_name, column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('content_content', 'content_topic', 'content_tutorpersona')
                    ORDER BY table_name, ordinal_position
                """)
                
                schema_info = "Database Schema:\n"
                current_table = None
                
                for row in cursor.fetchall():
                    table_name, column_name, data_type = row
                    
                    if current_table != table_name:
                        current_table = table_name
                        schema_info += f"\nTable: {table_name}\n"
                    
                    schema_info += f"  - {column_name}: {data_type}\n"
                
                return schema_info
                
        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            return "Schema information unavailable"

    def generate_sql_query(self, natural_language_query: str) -> str:
        """
        Generate SQL query from natural language using the database schema
        
        Args:
            natural_language_query: User's question in natural language
            
        Returns:
            Generated SQL query
        """
        try:
            # Validate input
            query = self._validate_text_input(natural_language_query, 1000)
            
            # Get database schema
            schema = self._get_database_schema()
            
            # Build prompt for SQL generation
            system_prompt = f"""You are a SQL query generator. Convert natural language questions to SQL queries.

{schema}

Rules:
- Only use tables and columns that exist in the schema
- Use proper SQL syntax for PostgreSQL
- Return only the SQL query, no explanations
- Use appropriate JOINs when referencing multiple tables
- Use LIMIT to prevent excessive results
- Be case-sensitive with column names"""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate SQL for: {query}"}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Basic validation - ensure it's a SELECT query
            if not sql_query.upper().startswith('SELECT'):
                raise ValueError("Generated query must be a SELECT statement")
            
            logger.info("Successfully generated SQL query")
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            raise Exception(f"Failed to generate SQL query: {str(e)}")

    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """
        Summarize long content into a shorter version
        
        Args:
            content: Text content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized content
        """
        try:
            # Validate input
            content = self._validate_text_input(content)
            
            # If content is already short enough, return as is
            if len(content) <= max_length:
                return content
            
            # Generate summary
            system_prompt = f"""Summarize the following content in {max_length} characters or less.
Keep the main points and key information. Be concise but informative."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=int(max_length / 3),  # Rough estimate for token count
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Successfully generated summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Content summarization failed: {str(e)}")
            raise Exception(f"Failed to summarize content: {str(e)}")

    def classify_content(self, content: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify content into provided categories with confidence scores
        
        Args:
            content: Text content to classify
            categories: List of possible categories
            
        Returns:
            Dictionary with category scores
        """
        try:
            # Validate inputs
            content = self._validate_text_input(content)
            if not categories:
                raise ValidationError("Categories list cannot be empty")
            
            categories_str = ", ".join(categories)
            
            system_prompt = f"""Classify the following content into these categories: {categories_str}

Return only a JSON object with category names as keys and confidence scores (0.0-1.0) as values.
The scores should sum to 1.0. Example: {{"category1": 0.7, "category2": 0.3}}"""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            classification = json.loads(result)
            
            # Validate that all categories are present
            for category in categories:
                if category not in classification:
                    classification[category] = 0.0
            
            logger.info("Successfully classified content")
            return classification
            
        except Exception as e:
            logger.error(f"Content classification failed: {str(e)}")
            # Return equal probability for all categories as fallback
            return {category: 1.0 / len(categories) for category in categories}

    def extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from content
        
        Args:
            content: Text content to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            # Validate input
            content = self._validate_text_input(content)
            
            system_prompt = f"""Extract the {max_keywords} most important keywords from the following content.
Return only the keywords separated by commas, no explanations."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            keywords_str = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_str.split(',')]
            
            logger.info(f"Successfully extracted {len(keywords)} keywords")
            return keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    def generate_questions(self, content: str, num_questions: int = 5, difficulty: str = "medium") -> List[str]:
        """
        Generate questions based on content for educational purposes
        
        Args:
            content: Source content for question generation
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            List of generated questions
        """
        try:
            # Validate inputs
            content = self._validate_text_input(content)
            if difficulty not in ["easy", "medium", "hard"]:
                difficulty = "medium"
            
            system_prompt = f"""Generate {num_questions} {difficulty} level questions based on the following content.
Make the questions educational and thought-provoking. Return only the questions, numbered."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            questions_text = response.choices[0].message.content.strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            logger.info(f"Successfully generated {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the LLM service
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "status": "healthy",
            "timestamp": logger.info,
            "components": {}
        }
        
        try:
            # Check OpenAI API
            test_response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            health_status["components"]["openai_api"] = "healthy"
            
        except Exception as e:
            health_status["components"]["openai_api"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check local embedding model
        if self.embedding_model:
            try:
                self.embedding_model.encode("test")
                health_status["components"]["local_embedding"] = "healthy"
            except Exception as e:
                health_status["components"]["local_embedding"] = f"unhealthy: {str(e)}"
        else:
            health_status["components"]["local_embedding"] = "not_available"
        
        # Check database connection
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status