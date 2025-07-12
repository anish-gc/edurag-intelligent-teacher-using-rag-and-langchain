from django.db import models

# Create your models here.

class QueryLog(models.Model):
    """General query logging for analytics"""
    QUERY_TYPE_CHOICES = [
        ('question', 'Question Answering'),
        ('nl_to_sql', 'Natural Language to SQL'),
        ('content_search', 'Content Search'),
        ('persona_chat', 'Persona Chat'),
    ]
    
    query = models.TextField()
    query_type = models.CharField(max_length=20, choices=QUERY_TYPE_CHOICES)
    response = models.JSONField()  # Can store different types of responses
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    processing_time = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'query_logs'
        indexes = [
            models.Index(fields=['query_type']),
            models.Index(fields=['created_at']),
            models.Index(fields=['session_id']),
        ]
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.get_query_type_display()} at {self.created_at}"


class SystemMetrics(models.Model):
    """System performance and usage metrics"""
    # Content metrics
    active_content_count = models.IntegerField(default=0)
    total_content_count = models.IntegerField(default=0)
    content_by_grade = models.JSONField(default=dict)
    content_by_topic = models.JSONField(default=dict)
    
    # Question metrics
    total_questions_answered = models.IntegerField(default=0)
    questions_by_type = models.JSONField(default=dict)
    average_response_time = models.FloatField(default=0)
    average_rating = models.FloatField(default=0)
    
    # Usage metrics
    topics_covered = models.IntegerField(default=0)
    most_active_persona = models.CharField(max_length=50, blank=True)
    persona_usage_stats = models.JSONField(default=dict)
    
    # Trends
    content_upload_trend = models.JSONField(default=list)
    question_trend = models.JSONField(default=list)
    rating_trend = models.JSONField(default=list)
    
    # Performance metrics
    avg_retrieval_time = models.FloatField(default=0)
    avg_llm_response_time = models.FloatField(default=0)
    total_tokens_used = models.BigIntegerField(default=0)
    
    collected_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'system_metrics'
        ordering = ['-collected_at']
        
    def __str__(self):
        return f"Metrics snapshot at {self.collected_at}"





class UserFeedback(models.Model):
    """Standalone feedback model for system improvement"""
    FEEDBACK_TYPE_CHOICES = [
        ('bug', 'Bug Report'),
        ('feature', 'Feature Request'),
        ('general', 'General Feedback'),
        ('content', 'Content Suggestion'),
    ]
    
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES)
    title = models.CharField(max_length=200)
    description = models.TextField()
    user_email = models.EmailField(blank=True)
    priority = models.CharField(
        max_length=10, 
        choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
        default='medium'
    )
    status = models.CharField(
        max_length=20,
        choices=[('open', 'Open'), ('in_progress', 'In Progress'), ('closed', 'Closed')],
        default='open'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'user_feedback'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.get_feedback_type_display()}: {self.title}"  



