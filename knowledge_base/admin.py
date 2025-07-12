from django.contrib import admin
from .models import Topic, Content, ContentTag, ContentTagging

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'parent_topic', 'is_active', 'created_at')
    list_filter = ('is_active', 'parent_topic')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at',)
    fieldsets = (
        (None, {
            'fields': ('name', 'description', 'parent_topic', 'is_active')
        }),
        ('Metadata', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    list_select_related = ('parent_topic',)

@admin.register(Content)
class ContentAdmin(admin.ModelAdmin):
    list_display = ('title', 'topic', 'grade', 'content_type', 'difficulty_level', 'is_active', 'is_verified')
    list_filter = ('grade', 'content_type', 'difficulty_level', 'is_active', 'is_verified', 'topic')
    search_fields = ('title', 'content_text', 'topic__name')
    readonly_fields = ('id', 'created_at', 'updated_at', 'last_accessed', 'view_count', 'retrieval_count', 'file_size')
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'topic', 'subtopic', 'grade', 'content_type', 'difficulty_level')
        }),
        ('Content Data', {
            'fields': ('content_text', 'file_path', 'file_type')
        }),
        ('Embeddings', {
            'fields': ('embedding_model', 'is_processed'),
            'classes': ('collapse',)
        }),
        ('Status & Verification', {
            'fields': ('is_active', 'is_verified', 'metadata')
        }),
        ('Analytics', {
            'fields': ('view_count', 'retrieval_count'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'last_accessed'),
            'classes': ('collapse',)
        }),
    )
    list_select_related = ('topic',)
    date_hierarchy = 'created_at'
    filter_horizontal = ()

    def get_queryset(self, request):
        return super().get_queryset(request).select_related('topic')

@admin.register(ContentTag)
class ContentTagAdmin(admin.ModelAdmin):
    list_display = ('name', 'color', 'description_short')
    search_fields = ('name', 'description')
    
    def description_short(self, obj):
        return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
    description_short.short_description = 'Description'

@admin.register(ContentTagging)
class ContentTaggingAdmin(admin.ModelAdmin):
    list_display = ('content_title', 'tag_name', 'created_at')
    list_filter = ('tag', 'created_at')
    search_fields = ('content__title', 'tag__name')
    readonly_fields = ('created_at',)
    
    def content_title(self, obj):
        return obj.content.title
    content_title.short_description = 'Content'
    
    def tag_name(self, obj):
        return obj.tag.name
    tag_name.short_description = 'Tag'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('content', 'tag')