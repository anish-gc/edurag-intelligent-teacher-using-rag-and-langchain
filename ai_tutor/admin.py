
from django.contrib import admin
from .models import TutorPersona, QuerySession, QuestionAnswer

@admin.register(TutorPersona)
class TutorPersonaAdmin(admin.ModelAdmin):
    list_display = ('display_name', 'name', 'tone', 'is_active', 'usage_count', 'average_rating')
    list_filter = ('is_active', 'name')
    search_fields = ('display_name', 'description', 'system_prompt')
    readonly_fields = ('usage_count', 'average_rating', 'created_at', 'updated_at')
    fieldsets = (
        (None, {
            'fields': ('name', 'display_name', 'description', 'is_active')
        }),
        ('LLM Configuration', {
            'fields': ('system_prompt', 'temperature', 'max_tokens', 'top_p')
        }),
        ('Persona Characteristics', {
            'fields': ('tone', 'response_style')
        }),
        ('Metrics', {
            'fields': ('usage_count', 'average_rating', 'created_at', 'updated_at')
        }),
    )

@admin.register(QuerySession)
class QuerySessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'persona', 'started_at', 'ended_at', 'total_queries')
    list_filter = ('persona', 'started_at')
    search_fields = ('session_id', 'user_ip', 'user_agent')
    readonly_fields = ('session_id', 'started_at', 'ended_at', 'total_queries')
    date_hierarchy = 'started_at'

@admin.register(QuestionAnswer)
class QuestionAnswerAdmin(admin.ModelAdmin):
    list_display = ('truncated_question', 'persona', 'question_type', 'created_at', 'rating')
    list_filter = ('persona', 'question_type', 'rating', 'created_at')
    search_fields = ('question', 'answer', 'feedback')
    readonly_fields = ('id', 'created_at', 'updated_at', 'processing_time', 'token_count')
    date_hierarchy = 'created_at'
    
    def truncated_question(self, obj):
        return obj.question[:50] + '...' if len(obj.question) > 50 else obj.question
    truncated_question.short_description = 'Question'