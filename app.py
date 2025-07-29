from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and label encoders
try:
    model = joblib.load("mdl.joblib")
    label_encoders = joblib.load("fixed_label_encoders.joblib")
    print("✅ Model and encoders loaded successfully")
    print(f"📊 Model expects {model.n_features_in_} features")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    label_encoders = None

def build_complete_features(user_input):
    """Build complete feature vector matching the trained model (43 features)"""
    
    # Extract user inputs with defaults
    city = user_input.get("city", "Mumbai")
    property_type = user_input.get("property_type", "Apartment")
    bhk = int(user_input.get("bedrooms", 2))
    bathrooms = int(user_input.get("bathrooms", 1))
    area_sqft = float(user_input.get("area_sqft", 1000))
    age = int(user_input.get("age", 5))
    floor = int(user_input.get("floor", 1))
    total_floors = int(user_input.get("total_floors", 10))
    furnishing = user_input.get("furnishing", "Unfurnished")
    parking = user_input.get("parking", "No")
    facing = user_input.get("facing", "North")
    
    # Safe encoding function
    def safe_encode(encoder_name, value):
        if encoder_name in label_encoders:
            encoder = label_encoders[encoder_name]
            if value in encoder.classes_:
                return encoder.transform([value])[0]
            else:
                print(f"⚠️ Unknown {encoder_name}: {value}, using default")
                return 0  # Default to first class
        return 0
    
    # Build feature vector based on typical dataset schema (43 features)
    features = []
    
    # 1. City (encoded)
    features.append(safe_encode("City", city))
    
    # 2. Locality (default encoded value)
    features.append(0)
    
    # 3. Property_Type (encoded)
    features.append(safe_encode("Property_Type", property_type))
    
    # 4. RERA_Approved (default: No = 0)
    features.append(0)
    
    # 5. BHK
    features.append(bhk)
    
    # 6. Bathrooms
    features.append(bathrooms)
    
    # 7. Balconies (estimate based on BHK)
    features.append(max(1, bhk - 1))
    
    # 8. Floor
    features.append(floor)
    
    # 9. Age_of_Property_years
    features.append(age)
    
    # 10. Ready_to_Move (default: No = 0)
    features.append(0)
    
    # 11. Furnishing (encoded)
    features.append(safe_encode("Furnishing", furnishing))
    
    # 12. Parking (encoded)
    features.append(safe_encode("Parking", parking))
    
    # 13. Facing (encoded)
    features.append(safe_encode("Facing", facing))
    
    # 14. Gated_Community (default: Yes = 1)
    features.append(1)
    
    # 15. Lift_Available (based on total floors)
    features.append(1 if total_floors > 3 else 0)
    
    # 16. Water_Supply (default encoded)
    features.append(0)
    
    # 17. Security_Guard (default: Yes = 1)
    features.append(1)
    
    # 18. Gym (default: No = 0)
    features.append(0)
    
    # 19. Swimming_Pool (default: No = 0)
    features.append(0)
    
    # 20. Power_Backup (default: Yes = 1)
    features.append(1)
    
    # 21. Clubhouse (default: No = 0)
    features.append(0)
    
    # 22. Play_Area (default: Yes = 1)
    features.append(1)
    
    # 23. Near_School_km
    features.append(2.5)
    
    # 24. Near_Hospital_km
    features.append(4.0)
    
    # 25. Near_Metro_km
    features.append(3.0)
    
    # 26. Near_Market_km
    features.append(1.5)
    
    # 27. Monthly_Maintenance (estimate based on area)
    features.append(area_sqft * 2.5)
    
    # 28. EMI_Per_Lakh
    features.append(1100.0)
    
    # 29. Interest_Rate
    features.append(8.5)
    
    # 30. Resale (default: Yes = 1)
    features.append(1)
    
    # 31. Property_Tax_Annual (estimate based on area)
    features.append(area_sqft * 12)
    
    # 32. Pollution_Index (city-based)
    city_pollution = {
        "Mumbai": 80, "Delhi": 90, "Bengaluru": 60, "Chennai": 70,
        "Hyderabad": 65, "Kolkata": 75, "Pune": 55, "Ahmedabad": 85
    }.get(city, 65)
    features.append(city_pollution)
    
    # 33. Noise_Index
    features.append(50.0)
    
    # 34. Crime_Rate
    features.append(15.0)
    
    # 35. Internet_Availability
    features.append(8.0)
    
    # 36. Public_Transport_Score
    features.append(7.5)
    
    # 37. Flood_Zone (default: No = 0)
    features.append(0)
    
    # 38. Earthquake_Zone (default: No = 0)
    features.append(0)
    
    # 39. Civic_Amenities_Rating
    features.append(7.0)
    
    # 40. Market_Demand_Rating
    features.append(6.5)
    
    # 41. Rental_Yield_Percent
    features.append(3.5)
    
    # 42. Carpet_Area_sqft
    features.append(area_sqft)
    
    # 43. Total_Floors
    features.append(total_floors)
    
    print(f"🔢 Built {len(features)} features for model")
    return np.array(features).reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        print("📩 Received data:", data)

        if not model or not label_encoders:
            return jsonify({"error": "Model not loaded properly"}), 500

        # Validate required fields
        city = data.get("city", "").strip()
        property_type = data.get("property_type", "").strip()
        
        if not city or not property_type:
            return jsonify({"error": "City and Property Type are required"}), 400

        # Validate numeric inputs
        try:
            area_sqft = float(data.get("area_sqft", 1000))
            if area_sqft <= 0:
                return jsonify({"error": "Area must be greater than 0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid area value"}), 400

        print(f"🏠 Processing: {city}, {property_type}, {area_sqft} sqft")

        # Build complete feature vector
        input_features = build_complete_features(data)
        
        if input_features is None:
            return jsonify({"error": "Failed to build feature vector"}), 400
            
        print(f"🔢 Input features shape: {input_features.shape}")
        
        # Verify feature count matches model expectation
        if input_features.shape[1] != model.n_features_in_:
            return jsonify({
                "error": f"Feature mismatch: got {input_features.shape[1]}, expected {model.n_features_in_}"
            }), 400

        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Ensure prediction is positive and reasonable
        if prediction < 0:
            prediction = abs(prediction)
        
        # Cap unrealistic predictions
        if prediction > 10000:  # More than 1 crore
            prediction = prediction / 10  # Adjust if needed
        
        # Format price in Indian currency
        price_lakhs = prediction
        price_formatted = f"₹ {price_lakhs:.2f} Lakhs"
        
        print(f"💰 Predicted price: {price_formatted}")

        return jsonify({
            "price": price_formatted,
            "price_lakhs": round(price_lakhs, 2),
            "details": {
                "city": city,
                "property_type": property_type,
                "area_sqft": area_sqft,
                "features_count": input_features.shape[1]
            }
        })

    except Exception as e:
        print("❌ Prediction Failed:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": label_encoders is not None,
        "expected_features": model.n_features_in_ if model else 0
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
