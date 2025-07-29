# House Price Predictor

A Flask web application for predicting house prices in major Indian cities using machine learning.

## Features

- Interactive map with city and locality selection
- Modern responsive UI with animations
- Real-time price predictions
- Support for 16+ major Indian cities
- Advanced property details input
- Mobile-friendly design

## Deployment on Render.com

### Prerequisites

Make sure you have these files in your repository:
- `mdl.joblib` (your trained model)
- `fixed_label_encoders.joblib` (your label encoders)

### Steps to Deploy

1. **Push to GitHub**: Make sure all files are in your GitHub repository

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service**:
   - **Name**: `house-price-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free tier is sufficient for testing

4. **Deploy**: Click "Create Web Service"

### Environment Variables

No additional environment variables are required for basic functionality.

### File Structure

\`\`\`
your-repo/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── render.yaml                     # Render configuration
├── runtime.txt                     # Python version
├── Procfile                        # Process file for deployment
├── templates/
│   └── index.html                  # Frontend template
├── mdl.joblib                      # Your ML model (add this)
├── fixed_label_encoders.joblib     # Your encoders (add this)
└── README.md                       # This file
\`\`\`

### Troubleshooting

1. **Build Fails**: Check that `requirements.txt` has all dependencies
2. **Model Not Found**: Ensure `mdl.joblib` and `fixed_label_encoders.joblib` are in root directory
3. **Port Issues**: The app automatically uses Render's PORT environment variable
4. **Static Files**: All CSS/JS is inline, no static file issues

### Local Development

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
\`\`\`

Visit `http://localhost:5000` to test locally.

### Support

- Check Render logs for deployment issues
- Ensure model files are not in `.gitignore`
- Verify all dependencies are in `requirements.txt`
