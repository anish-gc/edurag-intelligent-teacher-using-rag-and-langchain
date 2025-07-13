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
            text = text[:20000] if len(text) > 8000 else text

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

            try:
                results = self._pgvector_search(query_embedding, queryset, top_k)
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

    def _pg_vector(self, query_embedding, queryset, top_k):
        """Use pgvector for similarity search with VectorField"""
        with connection.cursor() as cursor:
            query_vector = query_embedding.tolist()

            # Get IDs from queryset
            content_ids = list(queryset.values_list("id", flat=True))

            if not content_ids:
                return []

            placeholders = ",".join(["%s"] * len(content_ids))
            sql = f"""
            SELECT c.id, c.title, t.name as topic_name, c.grade, c.content_text, 
                   c.subtopic, c.content_type, c.difficulty_level,
                   c.embedding <-> %s as distance
            FROM content_content c
            JOIN content_topic t ON c.topic_id = t.id
            WHERE c.id IN ({placeholders}) AND c.embedding IS NOT NULL
            ORDER BY distance ASC LIMIT %s
            """

            params = [query_vector] + content_ids + [top_k]
            cursor.execute(sql, params)
            results = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "topic": row[2],
                    "grade": row[3],
                    "content": row[4],
                    "subtopic": row[5],
                    "content_type": row[6],
                    "difficulty_level": row[7],
                    "similarity": max(0, 1 - row[8]),  # Convert distance to similarity
                }
                for row in results
            ]

    def _manual_similarity_search(self, query_embedding, queryset, top_k):
        """Manual cosine similarity calculation for VectorField"""
        results = []

        for content in queryset:
            try:
                if content.embedding:
                    # Handle different embedding storage formats
                    if hasattr(content.embedding, "tolist"):
                        content_embedding = np.array(content.embedding.tolist())
                    elif isinstance(content.embedding, (list, tuple)):
                        content_embedding = np.array(content.embedding)
                    else:
                        content_embedding = np.array(content.embedding)

                    # Ensure both embeddings have the same dimensions
                    if len(content_embedding) != len(query_embedding):
                        logger.warning(
                            f"Embedding dimension mismatch for content {content.id}"
                        )
                        continue

                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1), content_embedding.reshape(1, -1)
                    )[0][0]

                    results.append(
                        {
                            "id": content.id,
                            "title": content.title,
                            "topic": content.topic.name,
                            "grade": content.grade,
                            "content": content.content_text,
                            "subtopic": content.subtopic,
                            "content_type": content.content_type,
                            "difficulty_level": content.difficulty_level,
                            "similarity": float(similarity),
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing content {content.id}: {e}")
                continue

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

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
                        "id": content.id,
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

    def generate_answer(
        self, question, persona_name="friendly", grade_filter=None, topic_filter=None
    ):
        """Generate answer using RAG pipeline with comprehensive logging"""
        start_time = time.time()

        try:
            logger.info(f"Generating answer for question: {question[:100]}...")

            total_content = Content.objects.count()
            active_content = Content.objects.filter(is_active=True).count()

            content_with_embeddings = Content.objects.filter(
                embedding__isnull=False, is_active=True, is_processed=True
            ).count()
            logger.info(
                f"Content stats - Total: {total_content}, Active: {active_content}, With embeddings: {content_with_embeddings}"
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

            # Prepare context from retrieved content
            context = self._prepare_context(relevant_content)

            # Generate answer using LLM
            answer = self._call_llm(question, context, persona.system_prompt)

            # Calculate confidence based on similarity scores
            confidence = np.mean(
                [content["similarity"] for content in relevant_content]
            )
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
        """Call OpenAI API to generate answer with improved prompting"""
        try:
            full_prompt = f"""
            {system_prompt}
            
            You are helping a student with their educational question. Use the following context from educational materials to provide a helpful, accurate, and engaging answer.
            
            CONTEXT FROM EDUCATIONAL MATERIALS:
            {context}
            
            STUDENT'S QUESTION: {question}
            
            INSTRUCTIONS:
            1. Provide a clear, educational response that directly answers the student's question
            2. Use information from the context materials when relevant
            3. Explain concepts in a way appropriate for the student's level
            4. Be encouraging and supportive
            5. If the context doesn't fully answer the question, acknowledge this and provide what help you can
            6. Keep your response focused and not too lengthy
            
            Your response:
            """

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an educational tutor who provides clear, helpful explanations.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=600,
                temperature=0.7,
                top_p=0.9,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again later or rephrase your question."

    
    
    
    def create_retrieval_logs(self, question_answer, sources):
        """Create content retrieval logs for analytics"""
        try:
            for rank, source in enumerate(sources, 1):
                try:
                    content = Content.objects.get(id=source['id'])
                    ContentRetrievalLog.objects.create(
                        question_answer=question_answer,
                        content=content,
                        similarity_score=source.get('similarity', 0.0),
                        rank=rank
                    )
                    
                    # Update content retrieval count and last accessed
                    content.retrieval_count = models.F('retrieval_count') + 1
                    content.last_accessed = timezone.now()
                    content.save(update_fields=['retrieval_count', 'last_accessed'])
                    
                except Content.DoesNotExist:
                    logger.warning(f"Content with id {source['id']} not found for retrieval log")
                    continue
                    
        except Exception as e:
            logger.error(f"Error creating retrieval logs: {e}")
    
    def validate_embedding_consistency(self):
        """Validate embedding consistency across the database"""
        try:
            content_with_embeddings = Content.objects.filter(
                embedding__isnull=False,
                is_active=True
            )
            
            dimension_counts = {}
            invalid_embeddings = []
            
            for content in content_with_embeddings:
                try:
                    if hasattr(content.embedding, 'tolist'):
                        embedding_array = np.array(content.embedding.tolist())
                    else:
                        embedding_array = np.array(content.embedding)
                    
                    dimension = len(embedding_array)
                    dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
                    
                    # Check for invalid values
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        invalid_embeddings.append(content.id)
                        
                except Exception as e:
                    logger.error(f"Error validating embedding for content {content.id}: {e}")
                    invalid_embeddings.append(content.id)
            
            logger.info(f"Embedding validation complete. Dimensions: {dimension_counts}")
            if invalid_embeddings:
                logger.warning(f"Invalid embeddings found for content IDs: {invalid_embeddings}")
            
            return {
                'total_embeddings': content_with_embeddings.count(),
                'dimension_counts': dimension_counts,
                'invalid_embeddings': invalid_embeddings
            }
            
        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            return None