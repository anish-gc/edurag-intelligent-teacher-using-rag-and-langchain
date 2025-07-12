from django.contrib import admin
from .models import QueryLog, SystemMetrics, UserFeedback

@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ('query_short', 'query_type', 'processing_time', 'user_ip_short', 'created_at')
    list_filter = ('query_type', 'created_at')
    search_fields = ('query', 'response', 'session_id')
    readonly_fields = ('created_at', 'processing_time')
    date_hierarchy = 'created_at'
    
    def query_short(self, obj):
        return obj.query[:50] + '...' if len(obj.query) > 50 else obj.query
    query_short.short_description = 'Query'
    
    def user_ip_short(self, obj):
        return obj.user_ip if obj.user_ip else 'N/A'
    user_ip_short.short_description = 'IP'
    
    fieldsets = (
        ('Query Information', {
            'fields': ('query', 'query_type', 'response')
        }),
        ('User Information', {
            'fields': ('user_ip', 'user_agent', 'session_id'),
            'classes': ('collapse',)
        }),
        ('Performance', {
            'fields': ('processing_time',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )

@admin.register(SystemMetrics)
class SystemMetricsAdmin(admin.ModelAdmin):
    list_display = ('collected_at', 'total_questions_answered', 'average_rating', 'active_content_count')
    readonly_fields = ('collected_at',)
    date_hierarchy = 'collected_at'
    
    fieldsets = (
        ('Content Metrics', {
            'fields': ('active_content_count', 'total_content_count', 'content_by_grade', 'content_by_topic'),
            'classes': ('collapse',)
        }),
        ('Question Metrics', {
            'fields': ('total_questions_answered', 'questions_by_type', 'average_response_time', 'average_rating'),
            'classes': ('collapse',)
        }),
        ('Usage Metrics', {
            'fields': ('topics_covered', 'most_active_persona', 'persona_usage_stats'),
            'classes': ('collapse',)
        }),
        ('Trends', {
            'fields': ('content_upload_trend', 'question_trend', 'rating_trend'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('avg_retrieval_time', 'avg_llm_response_time', 'total_tokens_used'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('collected_at',),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        return False  # System metrics should be auto-generated

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ('title_short', 'feedback_type', 'priority', 'status', 'user_email_short', 'created_at')
    list_filter = ('feedback_type', 'priority', 'status', 'created_at')
    search_fields = ('title', 'description', 'user_email')
    list_editable = ('status', 'priority')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    
    def title_short(self, obj):
        return obj.title[:50] + '...' if len(obj.title) > 50 else obj.title
    title_short.short_description = 'Title'
    
    def user_email_short(self, obj):
        return obj.user_email[:15] + '...' if obj.user_email else 'Anonymous'
    user_email_short.short_description = 'Email'
    
    fieldsets = (
        ('Feedback Details', {
            'fields': ('feedback_type', 'title', 'description')
        }),
        ('User Information', {
            'fields': ('user_email',),
            'classes': ('collapse',)
        }),
        ('Management', {
            'fields': ('priority', 'status')
        }),
        ('Timestamps', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )