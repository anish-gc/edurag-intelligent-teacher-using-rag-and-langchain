import json
import logging
import time
from django.views import View
from django.http import JsonResponse

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import render
from django.utils import timezone
import uuid
from ai_tutor.models import QuerySession, QuestionAnswer, TutorPersona
from copying.rag_pipeline import RAGPipeline
from knowledge_base.models import Content
from retrieval.models import ContentRetrievalLog

logger = logging.getLogger(__name__)


def home_page(request):
    return render(request, "interactive_tutor_playground.html")


class BaseApiView(View):
    """Base class for API views with common functionality"""

    def dispatch(self, request, *args, **kwargs):
        """Add common headers and error handling"""
        try:
            return super().dispatch(request, *args, **kwargs)
        except Exception as e:
            logger.error(f"Unexpected error in {self.__class__.__name__}: {str(e)}")
            return JsonResponse(
                {
                    "error": "An unexpected error occurred",
                    "message": (
                        str(e)
                        if logger.isEnabledFor(logging.DEBUG)
                        else "Please try again later"
                    ),
                },
                status=500,
            )

    def parse_json_body(self, request):
        """Parse JSON body with error handling"""

        try:
            return json.loads(request.body)
        except json.JSONDecodeError:
            raise ValueError("The provided data has invalid json")

    def validate_pagination_params(self, request):
        try:
            page = int(request.GET.get("page", 1))
            page_size = min(int(request.GET.get("page_size", 20)), 100)
            return max(1, page), max(1, page_size)

        except ValueError:
            return 1, 20


@method_decorator(csrf_exempt, name="dispatch")
class AskQuestionView(BaseApiView):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_pipeline = RAGPipeline()

    def post(self, request):
        start_time = time.time()

        try:
            data = self.parse_json_body(request)
            question = data.get("question", "").strip()
            print(data)
            if not question:
                return JsonResponse({"error": "Question is required"}, status=400)

            if len(question) > 1000:
                return JsonResponse(
                    {"error": "Question is too long. Maximum 1000 characters."},
                    status=400,
                )

            # Extract parameters
            persona_name = data.get("persona", "friendly")
            grade_filter = data.get("grade")
            topic_filter = data.get("topic")
            session_id = data.get("session_id")

            # Get or create persona
            persona = self._get_or_create_persona(persona_name)
            
            # Get or create session
            session = self._get_or_create_session(session_id, persona, request)
            
            # Generate answer using RAG pipeline
            result = self.rag_pipeline.generate_answer(
                question=question,
                persona_name=persona_name,
                grade_filter=grade_filter,
                topic_filter=topic_filter
            )
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create QuestionAnswer record
            qa = self._create_question_answer(
                question=question,
                answer=result['answer'],
                persona=persona,
                session=session,
                result=result,
                processing_time=processing_time
            )
            # Log content retrievals
            self._log_content_retrievals(qa, result.get('sources', []))
              
            # Update session stats
            session.total_queries += 1
            session.save()
            
            # Format response
            response_data = {
                'id': qa.id,
                'question': question,
                'answer': result['answer'],
                'persona': persona_name,
                'confidence': result.get('confidence', 0.0),
                'relevant_sources': self._format_sources(result.get('sources', [])),
                'metadata': {
                    'grade_filter': grade_filter,
                    'topic_filter': topic_filter,
                    'sources_count': len(result.get('sources', [])),
                    'processing_time': round(processing_time, 3),
                    'session_id': str(session.session_id)
                }
            }
            
            return JsonResponse(response_data)
            
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Error in AskQuestionView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to process question',
                'message': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)  
      

    def _get_or_create_persona(self, persona_name):
        """Get or create persona with defaults"""

        persona, created = TutorPersona.objects.get_or_create(
            name=persona_name,
            defaults={
                "display_name": persona_name.capitalize(),
                "system_prompt": f"You are a {persona_name} and helpful educational tutor.",
                "description": f"{persona_name.capitalize()} tutoring style",
            },
        )
        return persona

    def _get_or_create_session(self, session_id, persona, request):
        """Get or create query session with more flexible session_id handling"""
        try:
            # First try to convert the incoming session_id to UUID if it's not already
            if session_id and not isinstance(session_id, uuid.UUID):
                try:
                    session_id = uuid.UUID(session_id)
                except (ValueError, AttributeError):
                    # If conversion fails, treat as new session
                    session_id = None
            
            if session_id:
                try:
                    session = QuerySession.objects.get(session_id=session_id)
                    return session
                except QuerySession.DoesNotExist:
                    pass
            
            # Create new session
            session = QuerySession.objects.create(
                persona=persona,
                user_ip=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:255]
            )
            return session
        except Exception as e:
            logger.error(f"Error in session handling: {str(e)}")
            # Fallback to creating new session
            return QuerySession.objects.create(
                persona=persona,
                user_ip=self._get_client_ip(request),
                user_agent=request.META.get('HTTP_USER_AGENT', '')[:255]
            )
    
    
    
    def _get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
    
    def _create_question_answer(self, question, answer, persona, session, result, processing_time):
        """Create QuestionAnswer record"""
        return QuestionAnswer.objects.create(
            question=question,
            answer=answer,
            persona=persona,
            session=session,
            retrieved_content=result.get('sources', []),
            retrieval_score=result.get('confidence', 0.0),
            retrieved_count=len(result.get('sources', [])),
            processing_time=processing_time,
            confidence_score=result.get('confidence', 0.0)
        )
        
    def _log_content_retrievals(self, qa, sources):
        """Log content retrieval for analytics"""
        for rank, source in enumerate(sources, 1):
            try:
                content = Content.objects.get(id=source['id'])
                ContentRetrievalLog.objects.create(
                    question_answer=qa,
                    content=content,
                    similarity_score=source.get('similarity', 0.0),
                    rank=rank
                )
                # Update content retrieval count
                content.retrieval_count += 1
                content.last_accessed = timezone.now()
                content.save(update_fields=['retrieval_count', 'last_accessed'])
            except Content.DoesNotExist:
                continue    
            
    def _format_sources(self, sources):
        """Format sources for frontend display"""
        formatted_sources = []
        
        for source in sources:
            formatted_source = {
                'id': source.get('id'),
                'title': source.get('title', 'Unknown'),
                'topic': source.get('topic', 'General'),
                'grade': source.get('grade', 'N/A'),
                'similarity': round(source.get('similarity', 0.0), 3),
                'excerpt': self._truncate_text(source.get('content', ''), 200)
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def _truncate_text(self, text, max_length):
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + '...'        