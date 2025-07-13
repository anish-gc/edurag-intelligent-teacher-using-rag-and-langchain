from django.shortcuts import render
import time
import json
import logging
from django.http import JsonResponse

from django.utils import timezone
from django.db import models
from ai_tutor.models import QuerySession, QuestionAnswer, TutorPersona
from ai_tutor.views import BaseApiView
from knowledge_base.models import Content, Topic
from monitoring.models import SystemMetrics

logger = logging.getLogger(__name__)

# Create your views here.


class GetMetricsView(BaseApiView):
    """Get comprehensive system metrics with caching"""

    def get(self, request):
        try:
            latest_metrics = SystemMetrics.objects.order_by("-collected_at").first()
            if (
                not latest_metrics
                or (timezone.now() - latest_metrics.collected_at).seconds > 3600
            ):
                latest_metrics = self._generate_metrics()
            # Basic counts
            total_content = Content.objects.count()
            total_questions = QuestionAnswer.objects.count()
            topics_count = Topic.objects.filter(is_active=True).count()
            # Advanced metrics
            avg_rating = (
                QuestionAnswer.objects.filter(rating__isnull=False).aggregate(
                    avg_rating=models.Avg("rating")
                )["avg_rating"]
                or 0.0
            )

            # Most popular topics (by content count)
            popular_topics = list(
                Content.objects.values("topic__name")
                .annotate(count=models.Count("id"))
                .order_by("-count")[:10]
            )

            # Most active topics (by question count)
            active_topics = list(
                QuestionAnswer.objects.values("retrieved_content")
                .annotate(count=models.Count("id"))
                .order_by("-count")[:5]
            )
            # Persona usage statistics
            persona_stats = list(
                QuestionAnswer.objects.values("persona__name", "persona__display_name")
                .annotate(
                    count=models.Count("id"),
                    avg_rating=models.Avg("rating"),
                    avg_confidence=models.Avg("confidence_score"),
                )
                .order_by("-count")
            )

            # Recent activity
            recent_questions = QuestionAnswer.objects.select_related(
                "persona"
            ).order_by("-created_at")[:10]

            # Performance metrics
            avg_processing_time = (
                QuestionAnswer.objects.filter(processing_time__isnull=False).aggregate(
                    avg=models.Avg("processing_time")
                )["avg"]
                or 0.0
            )

            return JsonResponse(
                {
                    "overview": {
                        "total_content_files": total_content,
                        "total_questions_answered": total_questions,
                        "topics_covered": topics_count,
                        "active_personas": TutorPersona.objects.filter(
                            is_active=True
                        ).count(),
                        "average_rating": round(avg_rating, 2),
                        "avg_processing_time": round(avg_processing_time, 3),
                    },
                    "content_stats": {
                        "content_with_embeddings": Content.objects.filter(
                            embedding__isnull=False
                        ).count(),
                        "content_without_embeddings": Content.objects.filter(
                            embedding__isnull=True
                        ).count(),
                        "verified_content": Content.objects.filter(
                            is_verified=True
                        ).count(),
                        "content_by_type": list(
                            Content.objects.values("content_type")
                            .annotate(count=models.Count("id"))
                            .order_by("-count")
                        ),
                    },
                    "popular_topics": popular_topics,
                    "persona_usage": persona_stats,
                    "recent_activity": [
                        {
                            "id": str(qa.id),
                            "question": self._truncate_text(qa.question, 100),
                            "persona": qa.persona.display_name,
                            "rating": qa.rating,
                            "confidence_score": qa.confidence_score,
                            "created_at": qa.created_at.isoformat(),
                        }
                        for qa in recent_questions
                    ],
                    "system_health": {
                        "avg_response_quality": avg_rating,
                        "total_sessions": QuerySession.objects.count(),
                        "avg_session_length": self._calculate_avg_session_length(),
                        "content_utilization": self._calculate_content_utilization(),
                    },
                    "last_updated": timezone.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error in GetMetricsView: {str(e)}")
            return JsonResponse({"error": "Failed to load metrics"}, status=500)

    def _generate_metrics(self):
        """Generate and store new metrics"""
        try:
            # This would typically be done in a background task
            metrics = SystemMetrics.objects.create(
                active_content_count=Content.objects.filter(is_active=True).count(),
                total_content_count=Content.objects.count(),
                total_questions_answered=QuestionAnswer.objects.count(),
                average_rating=QuestionAnswer.objects.filter(
                    rating__isnull=False
                ).aggregate(avg=models.Avg("rating"))["avg"]
                or 0.0,
                topics_covered=Topic.objects.filter(is_active=True).count(),
            )
            return metrics
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return None

    def _calculate_avg_session_length(self):
        """Calculate average session length"""
        try:
            sessions_with_end = QuerySession.objects.filter(ended_at__isnull=False)
            if sessions_with_end.exists():
                total_duration = sum(
                    [
                        (session.ended_at - session.started_at).total_seconds()
                        for session in sessions_with_end
                    ]
                )
                return total_duration / sessions_with_end.count()
            return 0
        except Exception as e:
            logger.error(f"Error calculating session length: {e}")
            return 0

    def _calculate_content_utilization(self):
        """Calculate what percentage of content is being actively used"""
        try:
            total_content = Content.objects.filter(is_active=True).count()
            utilized_content = Content.objects.filter(
                is_active=True, retrieval_count__gt=0
            ).count()

            if total_content > 0:
                return round((utilized_content / total_content) * 100, 2)
            return 0
        except Exception as e:
            logger.error(f"Error calculating content utilization: {e}")
            return 0

    def _truncate_text(self, text, max_length):
        """Truncate text with ellipsis"""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."
