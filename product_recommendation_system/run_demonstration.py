#!/usr/bin/env python3
import sys
import os
import numpy as np

# Add src to path
sys.path.append('src')

def simulate_authorized_transaction(simulator):
    """Simulate a successful transaction with authorized user"""
    print("🧪 SIMULATING AUTHORIZED TRANSACTION")
    print("=" * 50)
    
    # Use authorized user's data
    image_path = 'data/external/images/member1/neutral.jpg'
    audio_path = 'data/external/audio/member1/yes_approve.wav'
    
    # Enhanced user data based on your dataset features
    user_data = [
        82,    # engagement_score (from Twitter user)
        4.8,   # purchase_interest_score  
        3,     # social_media_platform_encoded (Twitter)
        1,     # review_sentiment_encoded (Neutral)
        332,   # purchase_amount
        4.2,   # customer_rating
        1,     # purchase_month (January)
        1,     # purchase_weekday (Monday)
        0.25,  # engagement_purchase_ratio
        1,     # high_engagement (1=True)
        0,     # high_purchase_interest (0=False) 
        82 * 4.8,  # engagement_interest_interaction
        332 * 4.2, # purchase_rating_interaction
        0,     # sentiment_rating_interaction
        3,     # platform_engagement_interaction
        1      # month_engagement_interaction
    ]
    
    recommendation = simulator.simulate_transaction(image_path, audio_path, user_data)
    return recommendation

def simulate_unauthorized_attempt(simulator):
    """Simulate an unauthorized attempt"""
    print("\n🚫 SIMULATING UNAUTHORIZED ATTEMPT")
    print("=" * 50)
    
    # Use unauthorized data
    image_path = 'data/external/images/unauthorized/unauthorized_face.jpg'
    audio_path = 'data/external/audio/unauthorized/unauthorized_voice.wav'
    
    # Generic user data
    user_data = [
        50,    # engagement_score
        2.0,   # purchase_interest_score  
        0,     # social_media_platform_encoded
        0,     # review_sentiment_encoded
        200,   # purchase_amount
        1.0,   # customer_rating
        1,     # purchase_month
        1,     # purchase_weekday
        0.1,   # engagement_purchase_ratio
        0,     # high_engagement (0=False)
        0,     # high_purchase_interest (0=False) 
        100,   # engagement_interest_interaction
        200,   # purchase_rating_interaction
        0,     # sentiment_rating_interaction
        0,     # platform_engagement_interaction
        0      # month_engagement_interaction
    ]
    
    recommendation = simulator.simulate_transaction(image_path, audio_path, user_data)
    return recommendation

def main():
    print("🚀 PRODUCT RECOMMENDATION SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    try:
        from system_simulation import SystemSimulator
        simulator = SystemSimulator('models')
        print("✅ System simulator initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing simulator: {e}")
        print("Please run main.py first to train the models.")
        return
    
    print("\n" + "=" * 60)
    
    # Successful transaction
    result1 = simulate_authorized_transaction(simulator)
    
    print("\n" + "=" * 60)
    
    # Unauthorized attempt  
    result2 = simulate_unauthorized_attempt(simulator)
    
    print("\n" + "=" * 60)
    print("🎯 DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    if result1:
        print(f"✅ Authorized transaction result: Recommendation = {result1}")
    if not result2:
        print("✅ Unauthorized attempt correctly blocked")
    else:
        print("❌ Unauthorized attempt was incorrectly approved")

if __name__ == "__main__":
    main()