# ACE Backend

Backend API built with Flask.

## Requirements

- Python 3.8+
- pip

## Instalation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Server

**Development:**
```bash
python app.py
```
The server will start on `http://localhost:5000`

**Production (Gunicorn, e.g. Render):**  
Use the config file so long-running requests (e.g. `/api/preview` Step0, up to ~4 min) are not killed by the worker timeout:

```bash
gunicorn -c gunicorn.conf.py app:app
```

If you cannot change the start command, set this env var instead:
```bash
GUNICORN_CMD_ARGS="--timeout 300 --graceful-timeout 300 --keep-alive 5"
```
Then start Gunicorn as usual (e.g. `gunicorn app:app`). Render will use `PORT` automatically.

## API Endpoints

### POST /api/generate

Generate an ad based on product information.

**Request:**
```json
{
  "productName": "string",
  "productDescription": "string"
}
```

**Success Response (200):**
```json
{
  "requestId": "uuid",
  "status": "success",
  "result": {
    "title": "string",
    "summary": "string",
    "files": []
  }
}
```

**Error Response (400/500):**
```json
{
  "status": "error",
  "message": "string"
}
```

### GET /health

Health check endpoint.

**Response (200):**
```json
{
  "status": "ok"
}
```

## CORS

CORS is enabled for localhost development (Vite frontend).

