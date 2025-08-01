# Generated by Django 5.2.4 on 2025-07-14 05:05

import django.core.validators
import django.db.models.deletion
import pgvector.django.vector
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TutorPersona',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('friendly', 'Friendly'), ('strict', 'Strict'), ('humorous', 'Humorous'), ('encouraging', 'Encouraging'), ('technical', 'Technical'), ('simplified', 'Simplified'), ('professional', 'Professional'), ('casual', 'Casual')], max_length=50, unique=True)),
                ('display_name', models.CharField(max_length=100)),
                ('system_prompt', models.TextField()),
                ('description', models.TextField(blank=True)),
                ('temperature', models.FloatField(default=0.7, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(2)])),
                ('max_tokens', models.IntegerField(default=500)),
                ('top_p', models.FloatField(default=1.0, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ('tone', models.CharField(blank=True, max_length=50)),
                ('response_style', models.CharField(blank=True, max_length=100)),
                ('is_active', models.BooleanField(default=True)),
                ('usage_count', models.PositiveIntegerField(default=0)),
                ('average_rating', models.FloatField(default=0.0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name_plural': 'Tutor Personas',
                'db_table': 'tutor_personas',
                'ordering': ['display_name'],
            },
        ),
        migrations.CreateModel(
            name='QuerySession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('session_id', models.UUIDField(default=uuid.uuid4, unique=True)),
                ('user_ip', models.GenericIPAddressField(blank=True, null=True)),
                ('user_agent', models.CharField(blank=True, max_length=255)),
                ('started_at', models.DateTimeField(auto_now_add=True)),
                ('ended_at', models.DateTimeField(blank=True, null=True)),
                ('total_queries', models.PositiveIntegerField(default=0)),
                ('persona', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ai_tutor.tutorpersona')),
            ],
            options={
                'db_table': 'query_sessions',
                'ordering': ['-started_at'],
            },
        ),
        migrations.CreateModel(
            name='QuestionAnswer',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('question', models.TextField()),
                ('question_type', models.CharField(choices=[('content', 'Content Question'), ('sql', 'Database Query'), ('general', 'General Question'), ('conceptual', 'Conceptual Question'), ('procedural', 'Procedural Question')], default='content', max_length=20)),
                ('query_embedding', pgvector.django.vector.VectorField(blank=True, dimensions=1536, null=True)),
                ('answer', models.TextField()),
                ('retrieved_content', models.JSONField(default=list)),
                ('retrieval_score', models.FloatField(blank=True, null=True)),
                ('retrieved_count', models.PositiveIntegerField(default=0)),
                ('confidence_score', models.FloatField(blank=True, null=True)),
                ('generated_sql', models.TextField(blank=True, null=True)),
                ('sql_execution_time', models.FloatField(blank=True, null=True)),
                ('processing_time', models.FloatField(blank=True, null=True)),
                ('token_count', models.PositiveIntegerField(blank=True, null=True)),
                ('llm_used', models.CharField(blank=True, max_length=50)),
                ('rating', models.IntegerField(blank=True, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(5)])),
                ('feedback', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('session', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='questions', to='ai_tutor.querysession')),
                ('persona', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='answers', to='ai_tutor.tutorpersona')),
            ],
            options={
                'db_table': 'question_answers',
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['session', 'created_at'], name='question_an_session_2721a8_idx'), models.Index(fields=['persona', 'created_at'], name='question_an_persona_36422b_idx'), models.Index(fields=['question_type'], name='question_an_questio_61b2aa_idx'), models.Index(fields=['rating'], name='question_an_rating_78bd67_idx'), models.Index(fields=['created_at'], name='question_an_created_f6ec8a_idx')],
            },
        ),
    ]
