# Banking Test Case Generator - Web Interface

A professional web application for generating test cases using your fine-tuned AI model.

## 🚀 Features

- **Modern UI**: Dark theme with gradient design
- **Real-time Generation**: Instant test case creation using AI
- **Multiple Test Types**: Functional, Security, Performance, Compliance
- **Template Mode**: Fast generation with consistent quality
- **Export Options**: Copy to clipboard or download as file
- **Responsive Design**: Works on all devices

## 🛠️ Tech Stack

- **Frontend**: Pure HTML, CSS, JavaScript (no frameworks)
- **Backend**: FastAPI (Python)
- **Model**: Your fine-tuned CodeT5 model (8 epochs, 8000+ samples)

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip
- Your trained model in `../model_trained/`

### Install Dependencies

```bash
pip install fastapi uvicorn transformers torch
```

## 🏃 Running the Application

### 1. Start the API Server

```bash
# From the project root directory
python -m uvicorn server:app --reload --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 2. Start the Web Server

```bash
# From the web directory
cd web
python -m http.server 8080
```

### 3. Open the Application

Open your browser and navigate to:
```
http://localhost:8080
```

## 💡 Usage

1. **Enter User Story**: Type your user story in the format "As a [user], I want to [action]"
2. **Select Test Type**: Choose from Functional, Security, Performance, or Compliance
3. **Adjust Settings**: 
   - Max Length: Control output length (50-200)
   - Template Mode: Enable for faster generation
4. **Generate**: Click the Generate button or press Ctrl+Enter
5. **Export**: Copy to clipboard or download the generated test case

## ⌨️ Keyboard Shortcuts

- **Ctrl + Enter**: Generate test case
- **Escape**: Clear form
- **Ctrl + C**: Copy result (when available)

## 📊 API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /api/model_info`: Model capabilities
- `POST /api/generate`: Generate test case
- `POST /api/generate_batch`: Batch generation

## 🎨 Customization

### Change Theme Colors

Edit `styles.css` and modify the CSS variables:

```css
:root {
    --primary: #667eea;
    --secondary: #764ba2;
    /* ... other colors */
}
```

### Modify API URL

Edit `app.js` and change the API_URL:

```javascript
const API_URL = 'http://localhost:8000';
```

## 🔧 Troubleshooting

### API Connection Error

If you see "Cannot connect to API server":
1. Ensure the API server is running
2. Check the port (default: 8000)
3. Verify no firewall blocking

### Model Loading Issues

If the model takes long to load:
1. First request loads the model (may take 30-60s)
2. Subsequent requests will be faster
3. Check `model_trained/` folder exists

## 📝 Sample User Stories

- "As a customer, I want to login using biometric authentication"
- "As a user, I want to transfer money between my accounts"
- "As a customer, I want to check my account balance"
- "As a user, I want to pay my utility bills"

## 🚢 Deployment

### Local Network

To access from other devices on your network:
1. Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
2. Start servers with `0.0.0.0` binding
3. Access via `http://YOUR_IP:8080`

### Production

For production deployment:
1. Use a proper ASGI server (Gunicorn + Uvicorn)
2. Set up NGINX as reverse proxy
3. Configure HTTPS with SSL certificates
4. Use environment variables for configuration

## 📄 License

This project is part of a Master's Thesis at HUIT.

## 👨‍💻 Author

**Vu Tuan Chien**  
Master's Thesis: Generative AI for Testing of Banking System in Mobile Applications  
HUIT - 2024
