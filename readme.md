# EduRAG: Intelligent Tutor Using RAG and LangChain

EduRAG is a backend-powered AI tutoring system built for Danson Solutions Pvt. Ltd. as part of a Python Developer Assessment. It leverages Retrieval-Augmented Generation (RAG), Large Language Model (LLM) APIs, and a PostgreSQL-based knowledge base to provide context-aware educational responses to student queries. The system is designed with a modular architecture, supporting content management, semantic retrieval, and natural language querying

## 🌟 Features

- **Content Upload & Knowledge Base**: Upload text-based content (e.g., .txt files) with metadata (topic, title, grade) and store in a PostgreSQL database.
- **Vector Embedding & Retrieval:**: Convert content into embeddings using LLM models and retrieve relevant content via semantic similarity with PGVector.
- **RAG-Based Question Answering:**: Retrieve relevant content using vector search and generate answers using an LLM.
- **Tutor Persona:**: Configurable AI tutor personas (e.g., friendly, strict, humorous) to customize response tone.
- **Natural Language SQL Querying**: Convert natural language queries (e.g., "What topics are covered in Grade 5?") into SQL for database interaction.
- **API Endpoints:**: RESTful API for content upload, question answering, topic retrieval, and system metrics.
- **Production-Ready Deployment:**: Deployed locally with Nginx as a reverse proxy and Gunicorn for production-like setup.

## 🔧 Technology Stack

- **Backend**: Django 5.2+
- **Database**: PostgreSQL 16+
- **Vector Store:**: pGVector
- **LLM APIs**: OpenAI (GPT) ,
- **Libraries**: LangChain, Django REST Framework ,
- **Deployment**: Gunicorn, Nginx
- **Frontend**: Basic HTML templates with CSS (static/styles.css)

## 📋 Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.12+
- PostgreSQL 16+
- Git
- pip
- virtualenv (optional for development)

## 🚀 Getting Started

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

```bash
git clone git@github.com:anish-gc/edurag-intelligent-teacher-using-rag-and-langchain.git
cd edurag-intelligent-teacher-using-rag-and-langchain
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements/development.txt
```

### 4. Configure PostgreSQL

Make sure PostgreSQL is installed and running. Create a database for the project:

```bash
# Access PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE edurag;

# Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Exit PostgreSQL
\q
```

### 5. Environment Variables

Create a `.env` file in the project root (You can take sample from .env-sample. Just copy all the contents to .env):

```
DEBUG=True
SECRET_KEY=your_secret_key_here
DB_NAME=edurag
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
OPENAI_API_KEY=your_openai_api_key
```

### 6. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Create a Superuser

```bash
python manage.py createsuperuser
```

### 8. Run the development Server

```bash
python manage.py runserver
```

The application should now be accessible at http://localhost:8000.

## 🗂️ Project Structure

```
edurag-intelligent-teacher-using-rag-and-langchain/
├── ai_tutor/              # Core app for RAG pipeline and LLM services
│   ├── llm_service_class.py  # LLM integration logic
│   ├── rag_pipeline.py      # RAG implementation
│   ├── models.py            # Database models for tutoring
│   ├── views.py             # API views
│   └── urls.py              # API routes
├── knowledge_base/         # App for content and metadata management
│   ├── models.py           # Database models for content
│   ├── views.py            # Content upload and management views
│   └── urls.py             # Content-related routes
├── monitoring/             # App for system metrics and stats
│   ├── models.py           # Metrics models
│   ├── views.py            # Metrics views
│   └── urls.py             # Metrics routes
├── retrieval/              # App for vector embedding and retrieval
│   ├── models.py           # Retrieval-related models
│   ├── views.py            # Retrieval views
│   └── urls.py             # Retrieval routes
├── edrag/                  # Django project settings
│   ├── settings.py         # Project configuration
│   ├── urls.py             # Root URL configuration
│   └── wsgi.py             # WSGI entry point
├── templates/              # HTML templates for web interface
│   ├── base.html           # Base template
│   ├── interactive_tutor_playground.html  # Tutor interaction UI
│   └── upload_content.html  # Content upload UI
├── static/                 # Static files
│   └── styles.css          # CSS styles
├── media/                  # Uploaded content files
│   └── content_files/      # Directory for uploaded .txt files
├── logs/                   # Log files
│   └── edurag.log          # Application logs
├── requirements/           # Dependencies
│   └── development.txt     # Development dependencies
├── gunicorn.conf.py        # Gunicorn configuration
└── manage.py               # Django management script
```

### 🔒 Local Deployment with Nginx

```bash
sudo apt update
sudo apt install nginx
```

### Configure Gunicorn as a Systemd Service

To ensure Gunicorn runs reliably, configure it as a systemd service:
Create a systemd service file at /etc/systemd/system/edurag.gunicorn.service:

```bash

[Unit]
Description=Gunicorn instance to serve Edurag app
After=network.target

[Service]
User=anishchengre
Group=anishchengre
WorkingDirectory=/home/anishchengre/Production-Ready/edurag-intelligent-teacher-using-rag-and-langchain
Environment="PATH=/home/anishchengre/Production-Ready/edurag-intelligent-teacher-using-rag-and-langchain/venv/bin"
ExecStart=/home/anishchengre/Production-Ready/edurag-intelligent-teacher-using-rag-and-langchain/venv/bin/gunicorn --config gunicorn.conf.py edrag.wsgi:application
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure

[Install]
WantedBy=multi-user.target

```

Replace anishchengre with your actual username and group, and ensure the WorkingDirectory and ExecStart paths match your project directory.

### Enable and start the Gunicorn service:
```bash
sudo systemctl enable edurag.gunicorn
sudo systemctl start edurag.gunicorn
```

### Verify the service is running
```bash

sudo systemctl status edurag.gunicorn

```

### Configure nginx

Create an Nginx configuration file (e.g., /etc/nginx/sites-available/edurag):

```bash
server {
    listen 80;
    server_name localhost;

    # Static files
    location /static/ {
        alias /home/anishchengre/Production-Ready/edurag-intelligent-teacher-using-rag-and-langchain/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Media files
    location /media/ {
        alias /home/anishchengre/Production-Ready/edurag-intelligent-teacher-using-rag-and-langchain/media/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_redirect off;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
}

```

### Enable Nginx Configuration

Link the configuration to Nginx's sites-enabled directory:

```bash
sudo ln -s /etc/nginx/sites-available/edurag /etc/nginx/sites-enabled/
```

### Test and Restart Nginx

```bash
# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```




## 📝 Development Guidelines

1. Follow PEP 8 coding style
2. Write tests for new features
3. Document code using docstrings
4. Use feature branches and pull requests

## 🔄 Deployment

For production deployment:

1. Set `DEBUG=False` in your .env file
2. Configure a proper web server (Nginx/Apache)
3. Use Gunicorn or uWSGI as the application server
4. Set up proper SSL certificates for all domains

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details

## 🆘 Troubleshooting

**Q: I get a "vector extension not found" error.**
A: Ensure the vector extension is enabled in PostgreSQL with CREATE EXTENSION IF NOT EXISTS vector;.

**Q:API endpoints return 404.**
A: Verify that urls.py is correctly configured and the Gunicorn service or development server is running.

**Q: Static files are not loading**
A: Run python manage.py collectstatic and ensure the Nginx configuration points to the correct static/ directory.

**Q: Gunicorn service fails to start.**
A: Check the service status with sudo systemctl status edurag.gunicorn and verify the paths in edurag.gunicorn.service.

## 📞 Support

For any questions or issues, please create an issue in the repository or contact project maintainers at anishgharti.chhetry@gmail.com.