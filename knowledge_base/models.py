from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from pgvector.django import VectorField

import uuid

class Topic(models.Model):
    """Normalized topic table for better data integrity"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    parent_topic = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='subtopics')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'topics'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Content(models.Model):
    """Enhanced content model with comprehensive metadata and tracking"""
    GRADE_CHOICES = [
        ('K', 'Kindergarten'),
        ('1', 'Grade 1'),
        ('2', 'Grade 2'),
        ('3', 'Grade 3'),
        ('4', 'Grade 4'),
        ('5', 'Grade 5'),
        ('6', 'Grade 6'),
        ('7', 'Grade 7'),
        ('8', 'Grade 8'),
        ('9', 'Grade 9'),
        ('10', 'Grade 10'),
        ('11', 'Grade 11'),
        ('12', 'Grade 12'),
        ('college', 'College'),
        ('general', 'General'),
    ]
    
    CONTENT_TYPE_CHOICES = [
        ('textbook', 'Textbook'),
        ('lecture', 'Lecture Notes'),
        ('exercise', 'Exercise'),
        ('example', 'Worked Example'),
        ('summary', 'Summary'),
        ('quiz', 'Quiz/Assessment'),
        ('reference', 'Reference Material'),
        ('other', 'Other'),
    ]

    DIFFICULTY_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255, db_index=True)
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='contents')
    subtopic = models.CharField(max_length=100, blank=True, null=True)
    grade = models.CharField(max_length=10, choices=GRADE_CHOICES, db_index=True)
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPE_CHOICES, default='textbook')
    difficulty_level = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='intermediate')
    
    # Content data
    content_text = models.TextField()
    file_path = models.FileField(upload_to='content_files/', null=True, blank=True)
    file_size = models.PositiveIntegerField(null=True, blank=True)  # in bytes
    file_type = models.CharField(max_length=10, default='txt')
    
    # Vector embedding for semantic search
    embedding = VectorField(dimensions=1536, null=True, blank=True)  # OpenAI embeddings
    embedding_model = models.CharField(max_length=50, default='text-embedding-3-small')
    
    # Metadata and status
    metadata = models.JSONField(default=dict)  # For additional unstructured metadata
    is_verified = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_processed = models.BooleanField(default=False)  # Track if embeddings are generated
    
    # Analytics
    view_count = models.PositiveIntegerField(default=0)
    retrieval_count = models.PositiveIntegerField(default=0)  # How many times retrieved in RAG
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_accessed = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'content'
        indexes = [
            models.Index(fields=['topic', 'grade']),
            models.Index(fields=['content_type']),
            models.Index(fields=['difficulty_level']),
            models.Index(fields=['is_active', 'is_processed']),
            models.Index(fields=['created_at']),
        ]
        ordering = ['topic', 'grade', 'title']
    
    def __str__(self):
        return f"{self.title} ({self.topic.name}, Grade {self.grade})"
    
    
class ContentTag(models.Model):
    """Tags for better content categorization"""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#007bff')  # Hex color code
    
    class Meta:
        db_table = 'content_tags'
        ordering = ['name']
        
    def __str__(self):
        return self.name


class ContentTagging(models.Model):
    """Many-to-many relationship between content and tags"""
    content = models.ForeignKey(Content, on_delete=models.CASCADE, related_name='tags')
    tag = models.ForeignKey(ContentTag, on_delete=models.CASCADE, related_name='content_items')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'content_tagging'
        unique_together = ['content', 'tag'] 
