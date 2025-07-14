import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from ai_tutor.llm_service_class import LLMService
from ai_tutor.models import QuerySession, QuestionAnswer, TutorPersona
from ai_tutor.views import BaseApiView
from knowledge_base.models import Content, Topic
from django.core.exceptions import ValidationError
logger = logging.getLogger(__name__)

from knowledge_base.models import Content

# Create your views here.
@method_decorator(csrf_exempt, name='dispatch')
class NaturalLanguageSQLView(BaseApiView):
    """Natural language to SQL conversion endpoint"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_service = LLMService()
    
    def post(self, request):
        try:
            data = self.parse_json_body(request)
            query = data.get('query', '').strip()
            
            if not query:
                return JsonResponse({'error': 'Query is required'}, status=400)
            
            # Convert natural language to SQL
            sql_query = self.llm_service.natural_language_to_sql(query)
            
            # Execute the SQL query safely
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append(dict(zip(columns, row)))
            
            return JsonResponse({
                'query': query,
                'sql_query': sql_query,
                'results': formatted_results,
                'count': len(formatted_results)
            })
            
        except ValidationError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Error in NaturalLanguageSQLView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to process query',
                'message': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class BatchEmbeddingView(BaseApiView):
    """Batch embedding generation for multiple content items"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_service = LLMService()
    
    def post(self, request):
        try:
            data = self.parse_json_body(request)
            content_ids = data.get('content_ids', [])
            
            if not content_ids:
                return JsonResponse({'error': 'Content IDs are required'}, status=400)
            
            # Get content items without embeddings
            content_items = Content.objects.filter(
                id__in=content_ids,
                embedding__isnull=True,
                is_active=True
            )
            
            if not content_items.exists():
                return JsonResponse({
                    'message': 'No content items found or all already have embeddings',
                    'processed': 0
                })
            
            # Extract texts
            texts = [item.content_text for item in content_items]
            
            # Generate embeddings in batch
            embeddings = self.llm_service.batch_generate_embeddings(texts)
            
            # Update content items with embeddings
            updated_count = 0
            for content_item, embedding in zip(content_items, embeddings):
                content_item.embedding = embedding
                content_item.is_processed = True
                content_item.save(update_fields=['embedding', 'is_processed'])
                updated_count += 1
            
            return JsonResponse({
                'message': f'Successfully generated embeddings for {updated_count} content items',
                'processed': updated_count,
                'total_requested': len(content_ids)
            })
            
        except ValidationError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            logger.error(f"Error in BatchEmbeddingView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to generate batch embeddings',
                'message': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)


class ServiceStatusView(BaseApiView):
    """Service health and status endpoint"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_service = LLMService()
    
    def get(self, request):
        try:
            # Get LLM service status
            service_status = self.llm_service.get_service_status()
            
            # Add additional system metrics
            service_status.update({
                'database_stats': {
                    'total_content': Content.objects.count(),
                    'content_with_embeddings': Content.objects.filter(embedding__isnull=False).count(),
                    'active_content': Content.objects.filter(is_active=True).count(),
                    'total_topics': Topic.objects.count(),
                    'total_personas': TutorPersona.objects.count(),
                    'total_sessions': QuerySession.objects.count(),
                    'total_questions': QuestionAnswer.objects.count(),
                }
            })
            
            return JsonResponse(service_status)
            
        except Exception as e:
            logger.error(f"Error in ServiceStatusView: {str(e)}")
            return JsonResponse({
                'service': 'LLMService',
                'status': 'unhealthy',
                'error': str(e)
            }, status=500)

