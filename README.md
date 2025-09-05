# FAQ Similarity Comparison App

A Flask application that compares different text similarity methods for FAQ matching.

## Features

- **Three Similarity Methods**: String-based cosine, semantic embeddings, and simple string matching
- **Dark Mode UI**: Modern, responsive interface
- **Test Questions**: Cycle through poorly worded questions to test methods
- **Real-time Comparison**: Side-by-side results with similarity scores

## Local Development

### Using Python directly
```bash
pip install -r requirements.txt
python app.py
```

### Using Docker
```bash
# Build and run with Docker Compose
docker-compose up

# Or build and run manually
docker build -t faq-app .
docker run -p 5000:5000 faq-app
```

## Deployment Options

### 1. Railway (Recommended - Free tier)
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### 2. Render (Free tier)

1. Push to GitHub
2. Connect repo at [render.com](https://render.com)
3. Select "Web Service" → "Docker"
4. Leave Build Command and Start Command empty
5. Enable Auto-Deploy

**Note**: Render automatically provides the `PORT` environment variable. No manual port configuration needed!

### 3. Fly.io (Free tier)
```bash
# Install flyctl first
fly auth login
fly launch
fly deploy
```

### 4. Heroku
```bash
heroku login
heroku create your-app-name
heroku stack:set container
git push heroku main
```

### 5. DigitalOcean App Platform
1. Push to GitHub
2. Create new app at DigitalOcean
3. Select Dockerfile build method
4. Set HTTP port to 5000

## Environment Variables

- `PORT`: Server port (default: 5000)
- `HOST`: Server host (default: 0.0.0.0)
- `FLASK_ENV`: Environment mode (development/production)

## Docker Configuration

The app is containerized with:
- Multi-stage build for optimization
- Non-root user for security
- Health checks
- Proper environment handling

## Tech Stack

- **Backend**: Flask, scikit-learn, sentence-transformers
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: Sentence transformers for semantic similarity
- **Deployment**: Docker, multiple platform support
