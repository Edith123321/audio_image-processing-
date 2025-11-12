#!/usr/bin/env python3
import sys
import os
import numpy as np

# Add src to path
sys.path.append('src')

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/external/images/member1/neutral.jpg',
        'data/external/images/member1/smiling.jpg', 
        'data/external/images/member2/neutral.jpg',
        'data/external/images/unauthorized/unauthorized_face.jpg',
        'data/external/audio/member1/yes_approve.wav',
        'data/external/audio/member2/yes_approve.wav',
        'data/external/audio/unauthorized/unauthorized_voice.wav'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print("✅ All required data files exist!")
    return True

def create_test_images():
    """Create test images if they don't exist"""
    import cv2
    import numpy as np
    
    # Create directories
    os.makedirs('data/external/images/member1', exist_ok=True)
    os.makedirs('data/external/images/member2', exist_ok=True) 
    os.makedirs('data/external/images/unauthorized', exist_ok=True)
    os.makedirs('data/external/audio/member1', exist_ok=True)
    os.makedirs('data/external/audio/member2', exist_ok=True)
    os.makedirs('data/external/audio/unauthorized', exist_ok=True)
    
    # Create simple test images
    for member in ['member1', 'member2']:
        for expression in ['neutral', 'smiling', 'surprised']:
            img_path = f'data/external/images/{member}/{expression}.jpg'
            if not os.path.exists(img_path):
                # Create a simple colored image
                img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                cv2.imwrite(img_path, img)
                print(f"✅ Created test image: {img_path}")
    
    # Create unauthorized face
    unauthorized_path = 'data/external/images/unauthorized/unauthorized_face.jpg'
    if not os.path.exists(unauthorized_path):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(unauthorized_path, img)
        print(f"✅ Created unauthorized image: {unauthorized_path}")

def create_test_audio():
    """Create test audio files if they don't exist"""
    import soundfile as sf
    import numpy as np
    
    # Sample rate
    sr = 22050
    
    for member in ['member1', 'member2']:
        for phrase in ['yes_approve', 'confirm_transaction']:
            audio_path = f'data/external/audio/{member}/{phrase}.wav'
            if not os.path.exists(audio_path):
                # Create simple audio (sine wave)
                duration = 2.0  # seconds
                t = np.linspace(0, duration, int(sr * duration))
                frequency = 440 if member == 'member1' else 550  # Different frequencies
                audio = 0.3 * np.sin(2 * np.pi * frequency * t)
                sf.write(audio_path, audio, sr)
                print(f"✅ Created test audio: {audio_path}")
    
    # Create unauthorized audio
    unauthorized_audio = 'data/external/audio/unauthorized/unauthorized_voice.wav'
    if not os.path.exists(unauthorized_audio):
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)  # Different frequency
        sf.write(unauthorized_audio, audio, sr)
        print(f"✅ Created unauthorized audio: {unauthorized_audio}")

def main():
    print("🛠️  SETTING UP SIMULATION ENVIRONMENT")
    print("=" * 50)
    
    # Create test data if needed
    print("📁 Creating test data files...")
    create_test_images()
    create_test_audio()
    
    # Check if data files exist
    if not check_data_files():
        print("❌ Please create the missing data files first")
        return
    
    print("\n✅ Simulation environment is ready!")
    print("🎮 Now run: python main.py")
    print("📊 Then run: python run_demonstration.py")

if __name__ == "__main__":
    main()