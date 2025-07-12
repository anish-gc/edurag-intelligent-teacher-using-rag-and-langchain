from django.contrib import admin
from .models import ContentRetrievalLog, NLToSQLQuery

@admin.register(ContentRetrievalLog)
class ContentRetrievalLogAdmin(admin.ModelAdmin):
    list_display = ('question_short', 'content_short', 'similarity_score', 'rank', 'was_helpful')
    list_filter = ('was_helpful', 'rank')
    search_fields = (
        'question_answer__question', 
        'question_answer__answer',
        'content__title',
        'content__content_text'
    )
    readonly_fields = ('similarity_score', 'rank',)
    list_select_related = ('question_answer', 'content')
    
    def question_short(self, obj):
        return obj.question_answer.question[:50] + '...' if obj.question_answer.question else ''
    question_short.short_description = 'Question'
    
    def content_short(self, obj):
        return obj.content.title[:50] + '...' if obj.content.title else ''
    content_short.short_description = 'Content'
    
    fieldsets = (
        ('Relationships', {
            'fields': ('question_answer', 'content')
        }),
        ('Retrieval Metrics', {
            'fields': ('similarity_score', 'rank')
        }),
        ('Feedback', {
            'fields': ('was_helpful',),
            'classes': ('collapse',)
        }),
    )

@admin.register(NLToSQLQuery)
class NLToSQLQueryAdmin(admin.ModelAdmin):
    list_display = ('query_short', 'execution_success', 'confidence_score', 'execution_time', 'created_at')
    list_filter = ('execution_success', 'question_answer__persona')
    search_fields = ('natural_language_query', 'generated_sql', 'error_message')
    readonly_fields = ('created_at', 'execution_time')
    list_select_related = ('question_answer',)
    
    def query_short(self, obj):
        return obj.natural_language_query[:50] + '...' if obj.natural_language_query else ''
    query_short.short_description = 'Natural Language Query'
    
    fieldsets = (
        ('Query Information', {
            'fields': ('natural_language_query', 'generated_sql', 'question_answer')
        }),
        ('Execution Results', {
            'fields': ('execution_result', 'execution_success', 'error_message')
        }),
        ('Performance Metrics', {
            'fields': ('confidence_score', 'execution_time'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('question_answer')