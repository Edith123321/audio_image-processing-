#!/usr/bin/env python3
import sys
import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append('src')

def check_models_exist():
    """Check if all required model files exist and are valid"""
    required_files = [
        'models/facial_recognition_model.pkl',
        'models/voice_verification_model.pkl', 
        'models/product_recommendation_model.pkl',
        'models/facial_scaler.pkl',
        'models/voice_scaler.pkl',
        'models/product_scaler.pkl'
    ]
    
    missing_files = []
    corrupted_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            # Check if file can be loaded
            try:
                joblib.load(file_path)
            except Exception as e:
                corrupted_files.append(f"{file_path} ({str(e)})")
    
    if missing_files:
        print("❌ Missing model files:")
        for f in missing_files:
            print(f"   - {f}")
    
    if corrupted_files:
        print("❌ Corrupted model files:")
        for f in corrupted_files:
            print(f"   - {f}")
    
    return len(missing_files) == 0 and len(corrupted_files) == 0

def create_fallback_simulator():
    """Create a fallback simulator when models are missing"""
    print("⚠️  Using fallback simulation mode (models not available)")
    
    class FallbackSimulator:
        def facial_authentication(self, image_path):
            print("🔍 Starting facial recognition...")
            print("✅ Facial authentication successful! (fallback)")
            return True, "member1"
        
        def voice_verification(self, audio_path):
            print("🎤 Starting voice verification...")
            print("✅ Voice verification successful! (fallback)")
            return True
        
        def product_recommendation(self, user_data):
            print("📊 Generating product recommendation...")
            recommendation = "Electronics"
            print(f"Recommended Product: {recommendation} (Confidence: 0.95)")
            return recommendation
        
        def simulate_transaction(self, image_path, audio_path, user_data):
            print("🚀 Starting transaction simulation...")
            print("=" * 50)
            
            # Step 1: Facial Recognition
            face_auth, user_id = self.facial_authentication(image_path)
            if not face_auth:
                print("❌ ACCESS DENIED: Facial recognition failed")
                return None
            
            # Step 2: Voice Verification
            voice_auth = self.voice_verification(audio_path)
            if not voice_auth:
                print("❌ ACCESS DENIED: Voice verification failed")
                return None
            
            # Step 3: Product Recommendation
            if face_auth and voice_auth:
                recommendation = self.product_recommendation(user_data)
                print(f"🎉 TRANSACTION APPROVED! Recommended product: {recommendation}")
                print("=" * 50)
                return recommendation
            else:
                print("❌ TRANSACTION DENIED!")
                print("=" * 50)
                return None
    
    return FallbackSimulator()

def simulate_authorized_transaction(simulator):
    """Simulate a successful transaction with authorized user"""
    print("🧪 SIMULATING AUTHORIZED TRANSACTION")
    print("=" * 50)
    
    # Use authorized user's data
    image_path = 'data/external/images/member1/neutral.jpg'
    audio_path = 'data/external/audio/member1/yes_approve.wav'
    
    # Sample user data for product recommendation (based on your dataset features)
    user_data = np.array([
        82,    # engagement_score (from Twitter user)
        4.8,   # purchase_interest_score  
        3,     # social_media_platform_encoded (Twitter)
        1,     # review_sentiment_encoded (Neutral)
        332,   # purchase_amount
        4.2,   # customer_rating
        1,     # purchase_month (January)
        1,     # purchase_weekday (Monday)
        0      # engagement_purchase_ratio (will be calculated)
    ])
    
    print("🔑 Starting authentication process...")
    recommendation = simulator.simulate_transaction(image_path, audio_path, user_data)
    return recommendation

def simulate_unauthorized_attempt(simulator):
    """Simulate an unauthorized attempt"""
    print("\n🚫 SIMULATING UNAUTHORIZED ATTEMPT")
    print("=" * 50)
    
    # Use unauthorized data
    image_path = 'data/external/images/unauthorized/unauthorized_face.jpg'
    audio_path = 'data/external/audio/unauthorized/unauthorized_voice.wav'
    
    # Sample user data
    user_data = np.array([
        50,    # engagement_score
        2.0,   # purchase_interest_score  
        0,     # social_media_platform_encoded
        0,     # review_sentiment_encoded
        200,   # purchase_amount
        1.0,   # customer_rating
        1,     # purchase_month
        1,     # purchase_weekday
        0      # engagement_purchase_ratio
    ])
    
    print("🔑 Starting authentication process...")
    recommendation = simulator.simulate_transaction(image_path, audio_path, user_data)
    return recommendation

def main():
    print("🚀 PRODUCT RECOMMENDATION SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    # Check if models exist and are valid
    if not check_models_exist():
        print("\n⚠️  Some model files are missing or corrupted.")
        print("Using fallback simulation mode...")
        simulator = create_fallback_simulator()
    else:
        print("✅ All model files are valid!")
        try:
            from system_simulation import SystemSimulator
            simulator = SystemSimulator('models')
            print("✅ System simulator initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing simulator: {e}")
            print("Using fallback simulation mode...")
            simulator = create_fallback_simulator()
    
    # Successful transaction
    result1 = simulate_authorized_transaction(simulator)
    
    print("\n" + "=" * 60)
    
    # Unauthorized attempt
    result2 = simulate_unauthorized_attempt(simulator)
    
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    if result1:
        print(f"✅ Authorized transaction result: {result1}")
    if not result2:
        print("✅ Unauthorized attempt correctly blocked")
    else:
        print("❌ Unauthorized attempt was incorrectly approved")

if __name__ == "__main__":
    main()