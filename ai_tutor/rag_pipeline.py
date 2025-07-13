import openai
from django.conf import settings
from sentence_transformers import SentenceTransformer
import time
import logging
import numpy as np
from django.db import connection
from django.db import models
from ai_tutor.models import TutorPersona
from knowledge_base.models import Content
from sklearn.metrics.pairwise import cosine_similarity
from django.utils import timezone
from retrieval.models import ContentRetrievalLog
import json
import uuid
import re

logger = logging.getLogger(__name__)


class RagPipeLine:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.default_personas = {
            "friendly": "You are a friendly and encouraging tutor who makes learning fun and accessible. Use simple language and positive reinforcement.",
            "strict": "You are a strict but fair tutor who emphasizes discipline and accuracy. Be precise and expect high standards.",
            "humorous": "You are a humorous tutor who uses jokes and fun examples to make learning enjoyable. Keep it light but educational.",
            "encouraging": "You are an encouraging tutor who builds confidence and celebrates progress. Focus on positive reinforcement and growth mindset.",
            "patient": "You are a patient tutor who takes time to explain concepts thoroughly. Break down complex topics into manageable parts.",
            "interactive": "You are an interactive tutor who engages students with questions and activities. Make learning participatory and engaging.",
        }

    def generate_embedding(self, text):
        """Generate embeddings using OpenAI API with fallback"""
        try:
            # Truncate text to avoid token limits
            text = text[:8000] if len(text) > 8000 else text

            response = openai.embeddings.create(
                model="text-embedding-ada-002", input=text
            )

            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            # Fallback to local model
            try:
                embedding = self.embedding_model.encode(text)
                return np.array(embedding)
            except Exception as e2:
                logger.error(f"Local embedding failed: {e2}")
                return None

    def semantic_search(self, query, top_k=5, grade_filter=None, topic_filter=None):
        """Perform semantic search with enhanced filtering and debugging"""
        try:
            start_time = time.time()

            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # Build queryset with filters
            queryset = Content.objects.filter(
                embedding__isnull=False, is_active=True, is_processed=True
            ).select_related("topic")

            # Apply filters
            if grade_filter:
                queryset = queryset.filter(grade__iexact=grade_filter)

            if topic_filter:
                # Filter by topic name or ID
                if topic_filter.isdigit():
                    queryset = queryset.filter(topic_id=topic_filter)
                else:
                    queryset = queryset.filter(topic__name__icontains=topic_filter)

            logger.info(
                f"Searching in {queryset.count()} content items with filters: grade={grade_filter}, topic={topic_filter}"
            )

            if queryset.count() == 0:
                logger.warning("No content with embeddings found for filters")
                return self._fallback_search(query, grade_filter, topic_filter)

            # Try pgvector first, then fallback to manual search
            try:
                results = self._pg_vector_search(query_embedding, queryset, top_k)
                logger.info(
                    f"pgvector search completed in {time.time() - start_time:.3f}s"
                )
                return results
            except Exception as e:
                logger.warning(
                    f"pgvector search failed: {e}, falling back to manual search"
                )
                results = self._manual_similarity_search(
                    query_embedding, queryset, top_k
                )
                logger.info(
                    f"Manual search completed in {time.time() - start_time:.3f}s"
                )
                return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return self._fallback_search(query, grade_filter, topic_filter)

    def _pg_vector_search(self, query_embedding, queryset, top_k):
        """Use pgvector for similarity search with proper vector casting"""
        with connection.cursor() as cursor:
            # Convert numpy array to list for PostgreSQL
            query_vector = query_embedding.tolist()
            content_ids = list(queryset.values_list("id", flat=True))

            if not content_ids:
                return []

            # Get table names
            table_name = Content._meta.db_table
            topic_table_name = Content._meta.get_field(
                "topic"
            ).related_model._meta.db_table

            # Create placeholders for IN clause
            placeholders = ",".join(["%s"] * len(content_ids))

            # Fixed SQL query with proper vector casting
            sql = f"""
                SELECT c.id, c.title, t.name as topic_name, c.grade, c.content_text, 
                       c.subtopic, c.content_type, c.difficulty_level,
                       c.embedding <-> %s::vector as distance
                FROM {table_name} c
                JOIN {topic_table_name} t ON c.topic_id = t.id
                WHERE c.id IN ({placeholders}) AND c.embedding IS NOT NULL
                ORDER BY distance ASC 
                LIMIT %s
            """

            # Prepare parameters: query_vector as JSON string, then content_ids, then top_k
            params = [json.dumps(query_vector)] + content_ids + [top_k]

            try:
                cursor.execute(sql, params)
                results = cursor.fetchall()

                formatted_results = []
                for row in results:
                    formatted_results.append({
                        "id": str(row[0]),  # Convert UUID to string
                        "title": row[1],
                        "topic": row[2],
                        "grade": row[3],
                        "content": row[4],
                        "subtopic": row[5],
                        "content_type": row[6],
                        "difficulty_level": row[7],
                        "similarity": max(0, 1 - float(row[8])),  # Convert distance to similarity
                    })
                
                return formatted_results
            except Exception as e:
                logger.error(f"pgvector query execution failed: {e}")
                # Re-raise to trigger fallback
                raise

    def _manual_similarity_search(self, query_embedding, queryset, top_k):
        """Manual cosine similarity calculation with proper error handling"""
        results = []

        # Convert query embedding to ensure it's a numpy array
        query_embedding = np.array(query_embedding)

        for content in queryset:
            try:
                if not content.embedding:
                    continue

                # Handle different embedding storage formats
                content_embedding = self._normalize_embedding(content.embedding)

                if content_embedding is None:
                    logger.warning(
                        f"Could not normalize embedding for content {content.id}"
                    )
                    continue

                # Ensure both embeddings have the same dimensions
                if len(content_embedding) != len(query_embedding):
                    logger.warning(
                        f"Embedding dimension mismatch for content {content.id}: "
                        f"content={len(content_embedding)}, query={len(query_embedding)}"
                    )
                    continue

                # Calculate cosine similarity - fix the array handling issue
                try:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1), content_embedding.reshape(1, -1)
                    )[0][0]

                    # Ensure similarity is a scalar value
                    if np.isscalar(similarity):
                        similarity_score = float(similarity)
                    else:
                        # If it's still an array, take the first element
                        similarity_score = float(
                            similarity.item()
                            if hasattr(similarity, "item")
                            else similarity[0]
                        )

                except Exception as sim_error:
                    logger.error(
                        f"Similarity calculation failed for content {content.id}: {sim_error}"
                    )
                    continue

                results.append(
                    {
                        "id": str(content.id),  # Convert UUID to string
                        "title": content.title,
                        "topic": content.topic.name,
                        "grade": content.grade,
                        "content": content.content_text,
                        "subtopic": content.subtopic,
                        "content_type": content.content_type,
                        "difficulty_level": content.difficulty_level,
                        "similarity": similarity_score,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing content {content.id}: {e}")
                continue

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def _normalize_embedding(self, embedding):
        """Normalize embedding to numpy array with better error handling"""
        try:
            # Handle Django VectorField - check if it has a specific method
            if hasattr(embedding, "tolist"):
                return np.array(embedding.tolist())
            elif hasattr(embedding, "__iter__") and not isinstance(embedding, str):
                # Handle list, tuple, or other iterables
                return np.array(list(embedding))
            elif isinstance(embedding, np.ndarray):
                return embedding
            elif isinstance(embedding, str):
                # Handle JSON string
                try:
                    return np.array(json.loads(embedding))
                except json.JSONDecodeError:
                    logger.error(
                        f"Invalid JSON in embedding string: {embedding[:100]}..."
                    )
                    return None
            else:
                logger.warning(f"Unknown embedding type: {type(embedding)}")
                return None
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            return None

    def _fallback_search(self, query, grade_filter, topic_filter):
        """Fallback to keyword search when embeddings fail"""
        try:
            logger.info("Using fallback keyword search")

            queryset = Content.objects.filter(is_active=True).select_related("topic")

            # Apply filters
            if grade_filter:
                queryset = queryset.filter(grade__iexact=grade_filter)

            if topic_filter:
                if topic_filter.isdigit():
                    queryset = queryset.filter(topic_id=topic_filter)
                else:
                    queryset = queryset.filter(topic__name__icontains=topic_filter)

            # Simple keyword search
            keywords = query.lower().split()
            for keyword in keywords:
                queryset = queryset.filter(
                    models.Q(title__icontains=keyword)
                    | models.Q(content_text__icontains=keyword)
                    | models.Q(subtopic__icontains=keyword)
                )

            results = []
            for content in queryset[:5]:  # Limit to top 5
                results.append(
                    {
                        "id": str(content.id),  # Convert UUID to string
                        "title": content.title,
                        "topic": content.topic.name,
                        "grade": content.grade,
                        "content": content.content_text,
                        "subtopic": content.subtopic,
                        "content_type": content.content_type,
                        "difficulty_level": content.difficulty_level,
                        "similarity": 0.5,  # Default similarity for keyword search
                    }
                )

            logger.info(f"Fallback search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    def _extract_specific_answer(self, question, content_text):
        """Extract specific answer from Q&A formatted content"""
        try:
            # Look for Q&A patterns in the content
            qa_patterns = [
                r"Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\n\s*(?:Q:|---)|\Z)",
                r"Question:\s*(.+?)\s*\n\s*Answer:\s*(.+?)(?=\n\s*(?:Question:|---)|\Z)",
                r"Q\d+[.)]\s*(.+?)\s*\n\s*A\d+[.)]\s*(.+?)(?=\n\s*(?:Q\d+|---)|\Z)",
            ]
            
            question_lower = question.lower().strip()
            
            for pattern in qa_patterns:
                matches = re.findall(pattern, content_text, re.MULTILINE | re.DOTALL)
                
                for q_match, a_match in matches:
                    q_clean = q_match.strip().lower()
                    a_clean = a_match.strip()
                    
                    # Check if this Q&A pair matches the user's question
                    if self._is_question_match(question_lower, q_clean):
                        return a_clean
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting specific answer: {e}")
            return None

    def _is_question_match(self, user_question, content_question):
        """Check if user question matches content question"""
        # Remove common question words and punctuation
        common_words = {'what', 'is', 'the', 'how', 'do', 'you', 'can', 'will', 'are', 'a', 'an'}
        
        def clean_question(q):
            q = re.sub(r'[^\w\s]', '', q.lower())
            words = q.split()
            return ' '.join([w for w in words if w not in common_words])
        
        user_clean = clean_question(user_question)
        content_clean = clean_question(content_question)
        
        # Check for exact match or high similarity
        if user_clean in content_clean or content_clean in user_clean:
            return True
        
        # Check for key terms match
        user_words = set(user_clean.split())
        content_words = set(content_clean.split())
        
        if user_words and content_words:
            overlap = len(user_words.intersection(content_words))
            return overlap / len(user_words) > 0.6
        
        return False

    def generate_answer(
        self, question, persona_name="friendly", grade_filter=None, topic_filter=None
    ):
        """Generate answer using RAG pipeline with improved content extraction"""
        start_time = time.time()

        try:
            logger.info(f"Generating answer for question: {question[:100]}...")

            # Content statistics
            total_content = Content.objects.count()
            active_content = Content.objects.filter(is_active=True).count()
            content_with_embeddings = Content.objects.filter(
                embedding__isnull=False, is_active=True, is_processed=True
            ).count()

            logger.info(
                f"Content stats - Total: {total_content}, Active: {active_content}, "
                f"With embeddings: {content_with_embeddings}"
            )

            # Get or create persona
            persona = self._get_or_create_persona(persona_name)

            # Retrieve relevant content
            relevant_content = self.semantic_search(
                question, top_k=5, grade_filter=grade_filter, topic_filter=topic_filter
            )

            logger.info(f"Found {len(relevant_content)} relevant pieces of content")

            if not relevant_content:
                # If no content found, provide a more helpful response
                answer = self._generate_general_answer(question, persona.system_prompt)
                processing_time = time.time() - start_time

                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": processing_time,
                    "method": "general_knowledge",
                }

            # Try to extract specific answer from the most relevant content
            best_content = relevant_content[0]
            extracted_answer = self._extract_specific_answer(question, best_content['content'])
            
            if extracted_answer and best_content['similarity'] > 0.4:
                # If we found a specific answer with good similarity, use it
                logger.info("Using extracted specific answer from knowledge base")
                
                # Format the extracted answer with persona style
                formatted_answer = self._format_extracted_answer(
                    extracted_answer, persona.system_prompt, question
                )
                
                confidence = float(best_content['similarity'])
                processing_time = time.time() - start_time

                return {
                    "answer": formatted_answer,
                    "sources": relevant_content[:3],
                    "confidence": confidence,
                    "processing_time": processing_time,
                    "method": "direct_extraction",
                }

            # Fallback to LLM generation with context
            context = self._prepare_context(relevant_content)
            answer = self._call_llm(question, context, persona.system_prompt)

            # Calculate confidence based on similarity scores
            confidence = np.mean([content["similarity"] for content in relevant_content])
            processing_time = time.time() - start_time

            # Update persona usage count
            persona.usage_count = models.F("usage_count") + 1
            persona.save(update_fields=["usage_count"])

            return {
                "answer": answer,
                "sources": relevant_content[:3],  # Top 3 sources
                "confidence": float(confidence),
                "processing_time": processing_time,
                "method": "rag_pipeline",
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            processing_time = time.time() - start_time

            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": processing_time,
                "method": "error_fallback",
            }

    def _format_extracted_answer(self, extracted_answer, system_prompt, question):
        """Format extracted answer with persona style"""
        try:
            # Simple formatting based on persona
            if "friendly" in system_prompt.lower():
                return f"{extracted_answer}\n\nI hope this helps! Feel free to ask if you need any clarification."
            elif "encouraging" in system_prompt.lower():
                return f"{extracted_answer}\n\nGreat question! You're doing well by asking about this concept."
            elif "strict" in system_prompt.lower():
                return f"{extracted_answer}\n\nMake sure you understand this formula thoroughly."
            else:
                return extracted_answer
                
        except Exception as e:
            logger.error(f"Error formatting extracted answer: {e}")
            return extracted_answer

    def _get_or_create_persona(self, persona_name):
        """Get or create persona with proper defaults"""
        try:
            persona = TutorPersona.objects.get(name=persona_name)
            return persona
        except TutorPersona.DoesNotExist:
            # Create default persona
            persona = TutorPersona.objects.create(
                name=persona_name,
                display_name=persona_name.capitalize(),
                system_prompt=self.default_personas.get(
                    persona_name, self.default_personas["friendly"]
                ),
                description=f"{persona_name.capitalize()} tutoring style - {self.default_personas.get(persona_name, 'General tutoring approach')}",
                is_active=True,
            )
            logger.info(f"Created new persona: {persona_name}")
            return persona

    def _generate_general_answer(self, question, system_prompt):
        """Generate a general answer when no specific content is found"""
        try:
            prompt = f"""
            {system_prompt}
            
            A student asked: "{question}"
            
            You don't have specific educational materials about this topic in your knowledge base, 
            but you can still provide a helpful general response. Provide a brief, educational answer 
            based on general knowledge and suggest how the student might learn more about this topic.
            
            Keep your response educational, encouraging, and helpful.
            """

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful educational tutor.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating general answer: {e}")
            return "I don't have specific information about that topic in my knowledge base. Could you provide more context or ask about a different topic? I'm here to help with your learning!"

    def _prepare_context(self, relevant_content):
        """Prepare context from relevant content with improved formatting"""
        context_parts = []

        for i, content in enumerate(relevant_content, 1):
            context_part = f"""
            Source {i}: {content['title']}
            - Grade Level: {content['grade']}
            - Topic: {content['topic']}
            - Subtopic: {content.get('subtopic', 'N/A')}
            - Content Type: {content.get('content_type', 'N/A')}
            - Difficulty: {content.get('difficulty_level', 'N/A')}
            - Relevance Score: {content['similarity']:.2f}

            Content:
            {content['content'][:1500]}...
            """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

    def _call_llm(self, question, context, system_prompt):
        """Call OpenAI API to generate answer with improved prompting for exact extraction"""
        try:
            full_prompt = f"""
            {system_prompt}
            
            You are helping a student with their educational question. The context below contains educational materials that may have the EXACT answer to the student's question.
            
            CONTEXT FROM EDUCATIONAL MATERIALS:
            {context}
            
            STUDENT'S QUESTION: {question}
            
            CRITICAL INSTRUCTIONS:
            1. First, look for EXACT matches between the student's question and any Q&A pairs in the context
            2. If you find an exact or very similar question in the context, use that exact answer
            3. If the context contains "Q: [question very similar to student's question]" followed by "A: [answer]", use that answer directly
            4. Only if no exact match is found, then provide a general educational response
            5. Be encouraging and supportive in your tone
            6. Keep your response focused and appropriate for the student's level
            
            Your response:
            """

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an educational tutor who prioritizes using exact answers from educational materials when available.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=600,
                temperature=0.3,  # Lower temperature for more consistent extraction
                top_p=0.9,
            )
            
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later or rephrase your question."

    def validate_embedding_consistency(self):
        """Validate embedding consistency across the database"""
        try:
            content_with_embeddings = Content.objects.filter(
                embedding__isnull=False, is_active=True
            )

            dimension_counts = {}
            invalid_embeddings = []

            for content in content_with_embeddings:
                try:
                    embedding_array = self._normalize_embedding(content.embedding)

                    if embedding_array is None:
                        invalid_embeddings.append(content.id)
                        continue

                    dimension = len(embedding_array)
                    dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

                    # Check for invalid values
                    if np.any(np.isnan(embedding_array)) or np.any(
                        np.isinf(embedding_array)
                    ):
                        invalid_embeddings.append(content.id)

                except Exception as e:
                    logger.error(
                        f"Error validating embedding for content {content.id}: {e}"
                    )
                    invalid_embeddings.append(content.id)

            logger.info(
                f"Embedding validation complete. Dimensions: {dimension_counts}"
            )
            if invalid_embeddings:
                logger.warning(
                    f"Invalid embeddings found for content IDs: {invalid_embeddings}"
                )

            return {
                "total_embeddings": content_with_embeddings.count(),
                "dimension_counts": dimension_counts,
                "invalid_embeddings": invalid_embeddings,
            }

        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            return None