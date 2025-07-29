# House Price Prediction Model

A machine learning-powered web application that predicts house prices based on various property features. This project combines data science techniques with a user-friendly web interface to provide accurate real estate price estimates.

## Overview

This project implements a comprehensive house price prediction system that leverages machine learning algorithms to estimate property values. The system processes various property characteristics such as location, size, amenities, and market conditions to provide accurate price predictions.

**Key Objectives:**
- Provide accurate house price predictions for buyers and sellers
- Offer an intuitive web interface for easy interaction
- Demonstrate end-to-end machine learning pipeline implementation
- Enable real-time price estimation based on property features

## Features

- **Price Prediction**: Advanced ML algorithms for accurate house price estimation
- **Web Interface**: User-friendly HTML templates for easy interaction
- **Data Processing**: Robust data preprocessing and feature engineering
- **Model Persistence**: Serialized models using joblib for quick loading
- **Responsive Design**: Mobile-friendly web interface
- **Real-time Predictions**: Instant price estimates upon feature input
- **Feature Engineering**: Advanced preprocessing with label encoders

## Technologies Used

### **Backend & ML**
- **Python** - Core programming language
- **scikit-learn** - Machine learning algorithms and preprocessing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **joblib** - Model serialization and persistence
- **Flask/Django** (inferred from app_updated.py) - Web framework

### **Frontend**
- **HTML5** - Web page structure and templates
- **CSS3** - Styling and responsive design
- **JavaScript** - Interactive frontend functionality

### **Development Tools**
- **Jupyter Notebook** - Data exploration and model development
- **Git** - Version control

## Models

The project implements multiple machine learning models for house price prediction:

### **Primary Models**
- **Linear Regression** - Baseline model for price prediction
- **Random Forest Regressor** - Ensemble method for improved accuracy

### **Model Features**
- **Feature Engineering**: Advanced preprocessing with label encoders
- **Cross-validation**: Robust model evaluation techniques
- **Hyperparameter Tuning**: Optimized model parameters
- **Model Persistence**: Serialized models (`mdl.joblib`) for production use

## Project Structure

```
House_Prices_Pred-Model/
│
├── templates/                 # HTML templates for web interface
│   ├── index.html            # Main prediction interface
│
├── Untitled.ipynb           # Jupyter notebook for model development
├── app_updated.py            # Main Flask/Django application
├── mdl.joblib               # Serialized trained model
├── fixed_label_encoders.joblib # Preprocessed label encoders
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Installation

### **Prerequisites**
- Python 3.7 or higher
- pip package manager

### **Setup Steps**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ajaysr01/House_Prices_Pred-Model.git
   cd House_Prices_Pred-Model
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app_updated.py
   ```

5. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## Usage

### **Web Interface**
1. Navigate to the main page
2. Input property features:
   - Location/Area
   - Property size (sq ft)
   - Number of bedrooms/bathrooms
   - Property age
   - Additional amenities
3. Click "Predict Price"
4. View the estimated price and confidence interval


## API Endpoints

- **GET /** - Main prediction interface
- **POST /predict** - Submit property data for price prediction
- **GET /results** - Display prediction results
- **GET /api/predict** - JSON API for programmatic access

## Model Performance
 
- **Root Mean Square Error (RMSE)**: Rs. 4034.80
- **R² Score**: 0.864

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author

**Ajaysr01**
- GitHub: [@Ajaysr01](https://github.com/Ajaysr01)

## Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Dataset providers for making real estate data available
- Contributors and testers who helped improve the model

---

⭐ **Star this repository if you found it helpful!**
