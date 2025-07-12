from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from pgvector.django import VectorField
import uuid


class TutorPersona(models.Model):
    """Enhanced tutor persona with LLM configuration"""
    PERSONA_CHOICES = [
        ('friendly', 'Friendly'),
        ('strict', 'Strict'),
        ('humorous', 'Humorous'),
        ('encouraging', 'Encouraging'),
        ('technical', 'Technical'),
        ('simplified', 'Simplified'),
        ('professional', 'Professional'),
        ('casual', 'Casual'),
    ]
    
    name = models.CharField(max_length=50, choices=PERSONA_CHOICES, unique=True)
    display_name = models.CharField(max_length=100)  # Human-readable name
    system_prompt = models.TextField()
    description = models.TextField(blank=True)
    
    # LLM Configuration
    temperature = models.FloatField(default=0.7, validators=[MinValueValidator(0), MaxValueValidator(2)])
    max_tokens = models.IntegerField(default=500)
    top_p = models.FloatField(default=1.0, validators=[MinValueValidator(0), MaxValueValidator(1)])
    
    # Persona characteristics
    tone = models.CharField(max_length=50, blank=True)
    response_style = models.CharField(max_length=100, blank=True)
    
    # Status and metrics
    is_active = models.BooleanField(default=True)
    usage_count = models.PositiveIntegerField(default=0)
    average_rating = models.FloatField(default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'tutor_personas'
        verbose_name_plural = 'Tutor Personas'
        ordering = ['display_name']
    
    def __str__(self):
        return self.display_name or self.get_name_display()


class QuerySession(models.Model):
    """Track user sessions for better analytics"""
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    persona = models.ForeignKey(TutorPersona, on_delete=models.CASCADE)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    total_queries = models.PositiveIntegerField(default=0)
    
    class Meta:
        db_table = 'query_sessions'
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Session {self.session_id} - {self.persona.display_name}"


class QuestionAnswer(models.Model):
    """Enhanced Q&A model with comprehensive tracking"""
    QUESTION_TYPE_CHOICES = [
        ('content', 'Content Question'),
        ('sql', 'Database Query'),
        ('general', 'General Question'),
        ('conceptual', 'Conceptual Question'),
        ('procedural', 'Procedural Question'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(QuerySession, on_delete=models.CASCADE, related_name='questions', null=True, blank=True)
    
    # Question details
    question = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPE_CHOICES, default='content')
    query_embedding = VectorField(dimensions=1536, null=True, blank=True)  # Embedding of the original question
    
    # Answer details
    answer = models.TextField()
    persona = models.ForeignKey(TutorPersona, on_delete=models.CASCADE, related_name='answers')
    
    # RAG-specific fields
    retrieved_content = models.JSONField(default=list)  # Store retrieved content IDs and scores
    retrieval_score = models.FloatField(null=True, blank=True)  # Average similarity score
    retrieved_count = models.PositiveIntegerField(default=0)
    confidence_score = models.FloatField(null=True, blank=True)  # LLM's confidence in the answer
    
    # SQL query tracking (for natural language SQL features)
    generated_sql = models.TextField(null=True, blank=True)
    sql_execution_time = models.FloatField(null=True, blank=True)
    
    # Performance metrics
    processing_time = models.FloatField(null=True, blank=True)  # Time to generate response
    token_count = models.PositiveIntegerField(null=True, blank=True)  # LLM token usage
    llm_used = models.CharField(max_length=50, blank=True)  # Which LLM was used
    
    # User feedback
    rating = models.IntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    feedback = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'question_answers'
        indexes = [
            models.Index(fields=['session', 'created_at']),
            models.Index(fields=['persona', 'created_at']),
            models.Index(fields=['question_type']),
            models.Index(fields=['rating']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Q: {self.question[:50]}... (Rating: {self.rating})"

