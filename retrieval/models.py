from django.db import models

from ai_tutor.models import QuestionAnswer
from knowledge_base.models import Content

# Create your models here.


class ContentRetrievalLog(models.Model):
    """Track which content is retrieved for which questions"""
    question_answer = models.ForeignKey(QuestionAnswer, on_delete=models.CASCADE, related_name='retrievals')
    content = models.ForeignKey(Content, on_delete=models.CASCADE, related_name='retrievals')
    similarity_score = models.FloatField()
    rank = models.PositiveIntegerField()  # Ranking in retrieval results
    was_helpful = models.BooleanField(null=True, blank=True)  # Optional feedback
    
    class Meta:
        db_table = 'content_retrieval_logs'
        unique_together = ['question_answer', 'content']
        indexes = [
            models.Index(fields=['question_answer', 'rank']),
            models.Index(fields=['content', 'similarity_score']),
        ]
    
class NLToSQLQuery(models.Model):
    """Natural Language to SQL query tracking"""
    natural_language_query = models.TextField()
    generated_sql = models.TextField()
    execution_result = models.JSONField()
    execution_success = models.BooleanField()
    error_message = models.TextField(blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    execution_time = models.FloatField(null=True, blank=True)
    
    # Link to question if part of Q&A
    question_answer = models.ForeignKey(QuestionAnswer, on_delete=models.CASCADE, null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'nl_to_sql_queries'
        verbose_name = 'Natural Language to SQL Query'
        verbose_name_plural = 'Natural Language to SQL Queries'
        indexes = [
            models.Index(fields=['execution_success']),
            models.Index(fields=['created_at']),
        ]
        
    def __str__(self):
        return f"NL Query: {self.natural_language_query[:50]}..."

    