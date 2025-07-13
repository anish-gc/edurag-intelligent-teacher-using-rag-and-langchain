from django.shortcuts import get_object_or_404, render
import json
import logging
from django.http import JsonResponse
from django.db import transaction
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import render
from ai_tutor.views import BaseApiView
from copying.llm_service_class import LLMService
from knowledge_base.models import Content, Topic
from django.db import models
from django.core.exceptions import ValidationError
logger = logging.getLogger(__name__)

from knowledge_base.models import Content

# Create your views here.

@method_decorator(csrf_exempt, name='dispatch')
class ContentUploadView(BaseApiView):
    """Enhanced content upload with comprehensive validation and processing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_service = LLMService()  # Initialize LLMService
    
    def get(self, request):
        """Render upload form"""
        return render(request, 'upload_content.html')
    
    def post(self, request):
        try:
            with transaction.atomic():
                # Extract and validate data
                file = request.FILES.get('file')
                if not file:
                    return JsonResponse({'error': 'File is required'}, status=400)
                
                # Validate file size (max 10MB)
                if file.size > 10 * 1024 * 1024:
                    return JsonResponse({
                        'error': 'File too large. Maximum size is 10MB.'
                    }, status=400)
                
                # Extract metadata
                title = request.POST.get('title', '').strip()
                topic_name = request.POST.get('topic', '').strip()
                grade = request.POST.get('grade', '').strip()
                content_type = request.POST.get('content_type', 'textbook')
                difficulty_level = request.POST.get('difficulty_level', 'intermediate')
                
                # Validate required fields
                if not all([title, topic_name, grade]):
                    return JsonResponse({
                        'error': 'Title, topic, and grade are required'
                    }, status=400)
                
                # Get or create topic
                topic, _ = Topic.objects.get_or_create(
                    name=topic_name,
                    defaults={'description': f'Auto-created topic for {topic_name}'}
                )
                
                # Read file content
                try:
                    content_text = file.read().decode('utf-8')
                except UnicodeDecodeError:
                    return JsonResponse({
                        'error': 'File must be UTF-8 encoded text'
                    }, status=400)
                
                # Validate content length
                if len(content_text) < 100:
                    return JsonResponse({
                        'error': 'Content is too short. Minimum 100 characters.'
                    }, status=400)
                
                # Create content record
                content = Content.objects.create(
                    title=title,
                    topic=topic,
                    grade=grade,
                    content_type=content_type,
                    difficulty_level=difficulty_level,
                    content_text=content_text,
                    file_path=file,
                    file_size=file.size,
                    file_type=file.name.split('.')[-1].lower() if '.' in file.name else 'txt'
                )
                
                # Generate embeddings using LLMService
                try:
                    embedding = self.llm_service.generate_embedding(content_text)
                    content.embedding = embedding
                    content.is_processed = True
                    content.save(update_fields=['embedding', 'is_processed'])
                    logger.info(f"Generated embedding for content {content.id}")
                except ValidationError as e:
                    logger.error(f"Validation error generating embedding for content {content.id}: {e}")
                    # Content is saved but without embedding
                except Exception as e:
                    logger.error(f"Failed to generate embedding for content {content.id}: {e}")
                    # Content is saved but without embedding
                
                return JsonResponse({
                    'message': 'Content uploaded successfully',
                    'content_id': str(content.id),
                    'has_embedding': content.embedding is not None,
                    'content_preview': content_text[:200] + '...' if len(content_text) > 200 else content_text
                }, status=201)
                
        except Exception as e:
            logger.error(f"Error in ContentUploadView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to upload content',
                'message': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)

            
            

class ContentListView(BaseApiView):
    """List content with advanced filtering and pagination"""
    
    def get(self, request):
        try:
            # Get pagination parameters
            page, page_size = self.validate_pagination_params(request)
            
            # Get filter parameters
            topic_filter = request.GET.get('topic')
            grade_filter = request.GET.get('grade')
            content_type_filter = request.GET.get('content_type')
            difficulty_filter = request.GET.get('difficulty')
            search = request.GET.get('search', '').strip()
            has_embedding = request.GET.get('has_embedding')
            
            # Build queryset
            queryset = Content.objects.select_related('topic').filter(is_active=True)
            
            # Apply filters
            if topic_filter:
                queryset = queryset.filter(topic__name__icontains=topic_filter)
            
            if grade_filter:
                queryset = queryset.filter(grade=grade_filter)
            
            if content_type_filter:
                queryset = queryset.filter(content_type=content_type_filter)
            
            if difficulty_filter:
                queryset = queryset.filter(difficulty_level=difficulty_filter)
            
            if search:
                queryset = queryset.filter(
                    models.Q(title__icontains=search) |
                    models.Q(content_text__icontains=search) |
                    models.Q(topic__name__icontains=search)
                )
            
            if has_embedding is not None:
                if has_embedding.lower() == 'true':
                    queryset = queryset.filter(embedding__isnull=False)
                elif has_embedding.lower() == 'false':
                    queryset = queryset.filter(embedding__isnull=True)
            
            # Pagination
            paginator = Paginator(queryset.order_by('-created_at'), page_size)
            page_obj = paginator.get_page(page)
            
            # Serialize data
            content_list = [
                {
                    'id': str(content.id),
                    'title': content.title,
                    'topic': content.topic.name,
                    'grade': content.grade,
                    'content_type': content.get_content_type_display(),
                    'difficulty_level': content.get_difficulty_level_display(),
                    'content_preview': self._truncate_text(content.content_text, 200),
                    'has_embedding': content.embedding is not None,
                    'view_count': content.view_count,
                    'retrieval_count': content.retrieval_count,
                    'file_size': content.file_size,
                    'created_at': content.created_at.isoformat(),
                    'updated_at': content.updated_at.isoformat()
                }
                for content in page_obj
            ]
            
            return JsonResponse({
                'content': content_list,
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_count': paginator.count,
                    'total_pages': paginator.num_pages,
                    'has_next': page_obj.has_next(),
                    'has_previous': page_obj.has_previous()
                },
                'filters': {
                    'topic': topic_filter,
                    'grade': grade_filter,
                    'content_type': content_type_filter,
                    'difficulty': difficulty_filter,
                    'search': search,
                    'has_embedding': has_embedding
                }
            })
            
        except Exception as e:
            logger.error(f"Error in ContentListView: {str(e)}")
            return JsonResponse({'error': 'Failed to load content'}, status=500)
    
    def _truncate_text(self, text, max_length):
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(' ', 1)[0] + '...'


class ContentDetailView(BaseApiView):
    """Get detailed view of specific content"""
    
    def get(self, request, content_id):
        try:
            content = get_object_or_404(Content, id=content_id)
            
            # Update view count
            content.view_count += 1
            content.save(update_fields=['view_count'])
            
            # Get related content
            related_content = Content.objects.filter(
                topic=content.topic,
                grade=content.grade,
                is_active=True
            ).exclude(id=content.id)[:5]
            
            return JsonResponse({
                'id': str(content.id),
                'title': content.title,
                'topic': content.topic.name,
                'subtopic': content.subtopic,
                'grade': content.grade,
                'content_type': content.get_content_type_display(),
                'difficulty_level': content.get_difficulty_level_display(),
                'content_text': content.content_text,
                'has_embedding': content.embedding is not None,
                'embedding_dimensions': len(content.embedding) if content.embedding else 0,
                'view_count': content.view_count,
                'retrieval_count': content.retrieval_count,
                'file_size': content.file_size,
                'file_type': content.file_type,
                'is_verified': content.is_verified,
                'created_at': content.created_at.isoformat(),
                'updated_at': content.updated_at.isoformat(),
                'related_content': [
                    {
                        'id': str(rc.id),
                        'title': rc.title,
                        'content_type': rc.get_content_type_display()
                    }
                    for rc in related_content
                ]
            })
            
        except Exception as e:
            logger.error(f"Error in ContentDetailView: {str(e)}")
            return JsonResponse({'error': 'Failed to load content details'}, status=500)


class GetTopicsView(BaseApiView):
    """Get topics filtered by grade"""
    
    def get(self, request):
        try:
            grade = request.GET.get('grade')
            
            # Build queryset
            queryset = Topic.objects.filter(is_active=True).distinct()
            
            # Filter topics that have content for the specified grade
            if grade:
                valid_grades = ['K'] + [str(i) for i in range(1, 13)] + ['college', 'general']
                if grade.upper() not in [g.upper() for g in valid_grades]:
                    return JsonResponse({
                        'error': 'Invalid grade. Must be K, 1-12, college, or general'
                    }, status=400)
                queryset = queryset.filter(contents__grade=grade, contents__is_active=True)
            
            # Serialize topics
            topics = [
                {
                    'id': topic.id,
                    'name': topic.name,
                    'description': topic.description,
                    'content_count': topic.contents.filter(is_active=True).count()
                }
                for topic in queryset
            ]
            
            return JsonResponse({
                'topics': topics,
                'count': len(topics),
                'grade_filter': grade
            })
            
        except Exception as e:
            logger.error(f"Error in GetTopicsView: {str(e)}")
            return JsonResponse({
                'error': 'Failed to load topics',
                'message': str(e) if logger.isEnabledFor(logging.DEBUG) else None
            }, status=500)            