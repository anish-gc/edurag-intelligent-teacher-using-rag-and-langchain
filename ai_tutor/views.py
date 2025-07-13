import json
import logging
import time
from django.views import View
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.core.paginator import Paginator
import uuid
    
from django.db import models

from ai_tutor.models import QuerySession, QuestionAnswer, TutorPersona
from ai_tutor.rag_pipeline import RagPipeLine
from knowledge_base.models import Content, Topic
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
        self.rag_pipeline = RagPipeLine()

    def post(self, request):
        start_time = time.time()

        try:
            data = self.parse_json_body(request)
            question = data.get("question", "").strip()
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

            # Validate grade
            if grade_filter and not self._is_valid_grade(grade_filter):
                return JsonResponse({
                    'error': 'Invalid grade. Must be K, 1-12, college, or general'
                }, status=400)
            
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
      

    def _is_valid_grade(self, grade):
        """Validate grade format"""
        valid_grades = ['K'] + [str(i) for i in range(1, 13)] + ['college', 'general']
        return str(grade).upper() in [g.upper() for g in valid_grades]
    
    
    def get(self, request):
        """Get available options and system status"""
        try:
            # Get available personas
            personas = TutorPersona.objects.filter(is_active=True)
            
            # Get available topics and grades
            topics = Topic.objects.filter(is_active=True).values('id', 'name', 'description')
            grades = Content.objects.values_list('grade', flat=True).distinct()
            
            # System status
            content_stats = {
                'total_content': Content.objects.count(),
                'content_with_embeddings': Content.objects.filter(embedding__isnull=False).count(),
                'active_content': Content.objects.filter(is_active=True).count()
            }
            
            return JsonResponse({
                'available_personas': [
                    {
                        'name': persona.name,
                        'display_name': persona.display_name,
                        'description': persona.description,
                        'usage_count': persona.usage_count
                    }
                    for persona in personas
                ],
                'available_topics': list(topics),
                'available_grades': sorted([g for g in grades if g]),
                'system_status': content_stats,
                'usage_guidelines': {
                    'max_question_length': 1000,
                    'supported_methods': ['POST', 'GET'],
                    'required_fields': ['question'],
                    'optional_fields': ['persona', 'grade', 'topic', 'session_id']
                }
            })
            
        except Exception as e:
            logger.error(f"Error in AskQuestionView GET: {str(e)}")
            return JsonResponse({'error': 'Failed to load options'}, status=500)

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class AnalyticsView(BaseApiView):
    """Advanced analytics and reporting for the tutoring system"""
    
    def get(self, request):
        try:
            # Get time range parameters
            days = int(request.GET.get('days', 30))
            start_date = timezone.now() - timezone.timedelta(days=days)
            
            # Usage trends over time
            usage_trends = list(
                QuestionAnswer.objects.filter(created_at__gte=start_date)
                .extra(select={'date': 'date(created_at)'})
                .values('date')
                .annotate(count=models.Count('id'))
                .order_by('date')
            )
            
            # Rating distribution
            rating_distribution = list(
                QuestionAnswer.objects.filter(
                    rating__isnull=False,
                    created_at__gte=start_date
                )
                .values('rating')
                .annotate(count=models.Count('id'))
                .order_by('rating')
            )
            
            # Topic performance
            topic_performance = list(
                QuestionAnswer.objects.filter(created_at__gte=start_date)
                .values('retrieved_content')
                .annotate(
                    count=models.Count('id'),
                    avg_rating=models.Avg('rating'),
                    avg_confidence=models.Avg('confidence_score'),
                    avg_processing_time=models.Avg('processing_time')
                )
                .order_by('-count')[:10]
            )
            
            # Persona effectiveness
            persona_effectiveness = list(
                QuestionAnswer.objects.filter(created_at__gte=start_date)
                .values('persona__name', 'persona__display_name')
                .annotate(
                    questions_answered=models.Count('id'),
                    avg_rating=models.Avg('rating'),
                    avg_confidence=models.Avg('confidence_score'),
                    avg_processing_time=models.Avg('processing_time')
                )
                .order_by('-questions_answered')
            )
            
            # Content retrieval patterns
            content_retrieval = list(
                ContentRetrievalLog.objects.filter(created_at__gte=start_date)
                .values('content__title', 'content__topic__name', 'content__grade')
                .annotate(
                    retrieval_count=models.Count('id'),
                    avg_similarity=models.Avg('similarity_score'),
                    avg_rank=models.Avg('rank')
                )
                .order_by('-retrieval_count')[:15]
            )
            
            # Grade-level distribution
            grade_distribution = list(
                QuestionAnswer.objects.filter(created_at__gte=start_date)
                .values('retrieved_content')
                .annotate(count=models.Count('id'))
                .order_by('retrieved_content')
            )
            
            # Session analytics
            session_analytics = {
                'total_sessions': QuerySession.objects.filter(started_at__gte=start_date).count(),
                'avg_questions_per_session': QuestionAnswer.objects.filter(
                    created_at__gte=start_date
                ).count() / max(QuerySession.objects.filter(started_at__gte=start_date).count(), 1),
                'bounce_rate': self._calculate_bounce_rate(start_date),
                'repeat_users': self._calculate_repeat_users(start_date)
            }
            
            return JsonResponse({
                'time_range': {
                    'days': days,
                    'start_date': start_date.isoformat(),
                    'end_date': timezone.now().isoformat()
                },
                'usage_trends': usage_trends,
                'rating_distribution': rating_distribution,
                'topic_performance': topic_performance,
                'persona_effectiveness': persona_effectiveness,
                'content_retrieval_patterns': content_retrieval,
                'grade_distribution': grade_distribution,
                'session_analytics': session_analytics,
                'summary_stats': {
                    'total_questions': QuestionAnswer.objects.filter(created_at__gte=start_date).count(),
                    'unique_users': QuerySession.objects.filter(started_at__gte=start_date).values('user_ip').distinct().count(),
                    'avg_satisfaction': QuestionAnswer.objects.filter(
                        created_at__gte=start_date,
                        rating__isnull=False
                    ).aggregate(avg=models.Avg('rating'))['avg'] or 0.0,
                    'content_coverage': self._calculate_content_coverage(start_date)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in AnalyticsView: {str(e)}")
            return JsonResponse({'error': 'Failed to load analytics'}, status=500)
    
    def _calculate_bounce_rate(self, start_date):
        """Calculate bounce rate (sessions with only one question)"""
        try:
            total_sessions = QuerySession.objects.filter(started_at__gte=start_date).count()
            single_question_sessions = QuerySession.objects.filter(
                started_at__gte=start_date,
                total_queries=1
            ).count()
            
            if total_sessions > 0:
                return round((single_question_sessions / total_sessions) * 100, 2)
            return 0
        except Exception as e:
            logger.error(f"Error calculating bounce rate: {e}")
            return 0
    
    def _calculate_repeat_users(self, start_date):
        """Calculate percentage of repeat users"""
        try:
            total_users = QuerySession.objects.filter(started_at__gte=start_date).values('user_ip').distinct().count()
            repeat_users = QuerySession.objects.filter(
                started_at__gte=start_date
            ).values('user_ip').annotate(
                session_count=models.Count('id')
            ).filter(session_count__gt=1).count()
            
            if total_users > 0:
                return round((repeat_users / total_users) * 100, 2)
            return 0
        except Exception as e:
            logger.error(f"Error calculating repeat users: {e}")
            return 0
    
    def _calculate_content_coverage(self, start_date):
        """Calculate what percentage of content was accessed"""
        try:
            total_content = Content.objects.filter(is_active=True).count()
            accessed_content = ContentRetrievalLog.objects.filter(
                created_at__gte=start_date
            ).values('content').distinct().count()
            
            if total_content > 0:
                return round((accessed_content / total_content) * 100, 2)
            return 0
        except Exception as e:
            logger.error(f"Error calculating content coverage: {e}")
            return 0
    
    
    
    


class QuestionHistoryView(BaseApiView):
    """Get question history with advanced filtering"""
    
    def get(self, request):
        try:
            # Get pagination parameters
            page, page_size = self.validate_pagination_params(request)
            
            # Get filter parameters
            persona_filter = request.GET.get('persona')
            rating_filter = request.GET.get('rating')
            session_id = request.GET.get('session_id')
            
            # Build queryset
            queryset = QuestionAnswer.objects.select_related('persona', 'session')
            
            # Apply filters
            if persona_filter:
                queryset = queryset.filter(persona__name=persona_filter)
            
            if rating_filter:
                try:
                    rating = int(rating_filter)
                    queryset = queryset.filter(rating=rating)
                except ValueError:
                    pass
            
            if session_id:
                queryset = queryset.filter(session__session_id=session_id)
            
            # Pagination
            paginator = Paginator(queryset.order_by('-created_at'), page_size)
            page_obj = paginator.get_page(page)
            
            # Serialize data
            questions = [
                {
                    'id': str(qa.id),
                    'question': qa.question,
                    'answer': self._truncate_text(qa.answer, 200),
                    'persona': qa.persona.name,
                    'rating': qa.rating,
                    'confidence_score': qa.confidence_score,
                    'processing_time': qa.processing_time,
                    'sources_count': len(qa.retrieved_content),
                    'created_at': qa.created_at.isoformat(),
                    'session_id': str(qa.session.session_id) if qa.session else None
                }
                for qa in page_obj
            ]
            
            return JsonResponse({
                'questions': questions,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': paginator.count,
                    'total_pages': paginator.num_pages,
                    'has_next': page_obj.has_next(),
                    'has_previous': page_obj.has_previous()
                },
                'filters': {
                    'persona': persona_filter,
                    'rating': rating_filter,
                    'session_id': session_id
                }
            })
            
        except Exception as e:
            logger.error(f"Error in QuestionHistoryView: {str(e)}")
            return JsonResponse({'error': 'Failed to load question history'}, status=500)
    
    def _truncate_text(self, text, max_length):
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + '...'

    

@method_decorator(csrf_exempt, name='dispatch')
class RateAnswerView(BaseApiView):
    """Rate and provide feedback on answers"""
    
    def post(self, request):
        try:
            data = self.parse_json_body(request)
            qa_id = data.get('qa_id')
            rating = data.get('rating')
            feedback = data.get('feedback', '').strip()
            
            # Validate rating
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                return JsonResponse({
                    'error': 'Rating must be an integer between 1 and 5'
                }, status=400)
            
            # Get and update QuestionAnswer
            qa = get_object_or_404(QuestionAnswer, id=qa_id)
            qa.rating = rating
            qa.feedback = feedback
            qa.save(update_fields=['rating', 'feedback'])
            
            # Update persona average rating
            persona = qa.persona
            avg_rating = QuestionAnswer.objects.filter(
                persona=persona,
                rating__isnull=False
            ).aggregate(avg=models.Avg('rating'))['avg']
            
            persona.average_rating = avg_rating or 0.0
            persona.save(update_fields=['average_rating'])
            
            return JsonResponse({
                'message': 'Rating saved successfully',
                'qa_id': qa_id,
                'rating': rating,
                'feedback': feedback
            })
            
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Error in RateAnswerView: {str(e)}")
            return JsonResponse({'error': 'Failed to save rating'}, status=500)
    
    
    
    

class ExportDataView(BaseApiView):
    """Export data for analysis or backup"""
    
    def get(self, request):
        try:
            export_type = request.GET.get('type', 'questions')
            format_type = request.GET.get('format', 'json')
            days = int(request.GET.get('days', 30))
            
            start_date = timezone.now() - timezone.timedelta(days=days)
            
            if export_type == 'questions':
                data = self._export_questions(start_date)
            elif export_type == 'content':
                data = self._export_content()
            elif export_type == 'sessions':
                data = self._export_sessions(start_date)
            elif export_type == 'analytics':
                data = self._export_analytics(start_date)
            else:
                return JsonResponse({'error': 'Invalid export type'}, status=400)
            
            if format_type == 'json':
                response = JsonResponse(data)
                response['Content-Disposition'] = f'attachment; filename="{export_type}_export.json"'
                return response
            else:
                return JsonResponse({'error': 'Only JSON format is currently supported'}, status=400)
                
        except Exception as e:
            logger.error(f"Error in ExportDataView: {str(e)}")
            return JsonResponse({'error': 'Failed to export data'}, status=500)
    
    def _export_questions(self, start_date):
        """Export question-answer data"""
        questions = QuestionAnswer.objects.filter(
            created_at__gte=start_date
        ).select_related('persona', 'session')
        
        return {
            'export_type': 'questions',
            'exported_at': timezone.now().isoformat(),
            'count': questions.count(),
            'data': [
                {
                    'id': str(qa.id),
                    'question': qa.question,
                    'answer': qa.answer,
                    'persona': qa.persona.name,
                    'rating': qa.rating,
                    'feedback': qa.feedback,
                    'confidence_score': qa.confidence_score,
                    'processing_time': qa.processing_time,
                    'retrieved_count': qa.retrieved_count,
                    'created_at': qa.created_at.isoformat(),
                    'session_id': str(qa.session.session_id) if qa.session else None
                }
                for qa in questions
            ]
        }
    
    def _export_content(self):
        """Export content data"""
        content = Content.objects.filter(is_active=True).select_related('topic')
        
        return {
            'export_type': 'content',
            'exported_at': timezone.now().isoformat(),
            'count': content.count(),
            'data': [
                {
                    'id': str(c.id),
                    'title': c.title,
                    'topic': c.topic.name,
                    'subtopic': c.subtopic,
                    'grade': c.grade,
                    'content_type': c.content_type,
                    'difficulty_level': c.difficulty_level,
                    'has_embedding': c.embedding is not None,
                    'view_count': c.view_count,
                    'retrieval_count': c.retrieval_count,
                    'file_size': c.file_size,
                    'created_at': c.created_at.isoformat(),
                    'updated_at': c.updated_at.isoformat()
                }
                for c in content
            ]
        }
    
    def _export_sessions(self, start_date):
        """Export session data"""
        sessions = QuerySession.objects.filter(started_at__gte=start_date).select_related('persona')
        
        return {
            'export_type': 'sessions',
            'exported_at': timezone.now().isoformat(),
            'count': sessions.count(),
            'data': [
                {
                    'session_id': str(s.session_id),
                    'persona': s.persona.name,
                    'total_queries': s.total_queries,
                    'user_ip': s.user_ip,
                    'started_at': s.started_at.isoformat(),
                    'ended_at': s.ended_at.isoformat() if s.ended_at else None,
                    'duration_seconds': (s.ended_at - s.started_at).total_seconds() if s.ended_at else None
                }
                for s in sessions
            ]
        }
    
    def _export_analytics(self, start_date):
        """Export analytics summary"""
        return {
            'export_type': 'analytics',
            'exported_at': timezone.now().isoformat(),
            'time_range': {
                'start_date': start_date.isoformat(),
                'end_date': timezone.now().isoformat()
            },
            'summary': {
                'total_questions': QuestionAnswer.objects.filter(created_at__gte=start_date).count(),
                'total_sessions': QuerySession.objects.filter(started_at__gte=start_date).count(),
                'unique_users': QuerySession.objects.filter(started_at__gte=start_date).values('user_ip').distinct().count(),
                'avg_rating': QuestionAnswer.objects.filter(
                    created_at__gte=start_date,
                    rating__isnull=False
                ).aggregate(avg=models.Avg('rating'))['avg'] or 0.0,
                'content_accessed': ContentRetrievalLog.objects.filter(
                    created_at__gte=start_date
                ).values('content').distinct().count()
            }
        }
            
            
            
            


