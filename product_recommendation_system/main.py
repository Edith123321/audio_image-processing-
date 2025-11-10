import os
import sys
import importlib
import pandas as pd
import numpy as np

# Check and install required packages
required_packages = {
    'cv2': 'opencv-python',
    'librosa': 'librosa',  
    'sklearn': 'scikit-learn',
    'xgboost': 'xgboost',
    'soundfile': 'soundfile'
}

def check_dependencies():
    """Check if all required packages are installed"""
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

def check_data_files():
    """Check if required data files exist"""
    required_files = {
        'customer_social_profiles.csv': 'data/raw/customer_social_profiles.csv',
        'customer_transactions.csv': 'data/raw/customer_transactions.csv'
    }
    
    missing_files = []
    for file_name, file_path in required_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_name} found")
        else:
            print(f"❌ {file_name} missing")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        print("Please ensure the CSV files are in the data/raw/ directory")
        return False
    
    return True

def load_and_validate_data():
    """Load and validate the existing CSV files"""
    print("\nLoading and validating data files...")
    
    try:
        # Load social profiles data
        social_df = pd.read_csv('data/raw/customer_social_profiles.csv')
        print(f"✅ Social profiles loaded: {social_df.shape[0]} rows, {social_df.shape[1]} columns")
        print("Social profiles columns:", list(social_df.columns))
        
        # Load transactions data
        transactions_df = pd.read_csv('data/raw/customer_transactions.csv')
        print(f"✅ Transactions loaded: {transactions_df.shape[0]} rows, {transactions_df.shape[1]} columns")
        print("Transactions columns:", list(transactions_df.columns))
        
        # Display sample data
        print("\n📊 Social Profiles Sample:")
        print(social_df.head(3))
        print("\n📊 Transactions Sample:")
        print(transactions_df.head(3))
        
        return social_df, transactions_df
        
    except Exception as e:
        print(f"❌ Error loading data files: {e}")
        return None, None

def create_dummy_features():
    """Create dummy feature files for testing when real image/audio data isn't available"""
    print("\nCreating dummy feature files for model training...")
    
    # Create dummy image features
    image_features = []
    members = ['member1', 'member2', 'member3']
    expressions = ['neutral', 'smiling', 'surprised']
    augmentations = ['original', 'rotated_15', 'flipped', 'grayscale', 'brightened']
    
    for member in members:
        for expr in expressions:
            for aug in augmentations:
                features = {
                    'member_id': member,
                    'image_file': f'{expr}.jpg',
                    'augmentation': aug,
                    'expression': expr,
                    'hist_r_mean': np.random.uniform(50, 150),
                    'hist_r_std': np.random.uniform(10, 30),
                    'hist_g_mean': np.random.uniform(50, 150),
                    'hist_g_std': np.random.uniform(10, 30),
                    'hist_b_mean': np.random.uniform(50, 150),
                    'hist_b_std': np.random.uniform(10, 30),
                    'hog_mean': np.random.uniform(100, 200),
                    'hog_std': np.random.uniform(20, 40),
                    'edge_density': np.random.uniform(0.1, 0.5),
                    'texture_mean': np.random.uniform(100, 200),
                    'texture_std': np.random.uniform(20, 40)
                }
                image_features.append(features)
    
    image_df = pd.DataFrame(image_features)
    image_df.to_csv('data/processed/image_features.csv', index=False)
    print("✅ Dummy image features created!")
    
    # Create dummy audio features
    audio_features = []
    phrases = ['yes_approve', 'confirm_transaction']
    
    for member in members:
        for phrase in phrases:
            for aug in augmentations[:3]:  # Fewer audio augmentations
                features = {
                    'member_id': member,
                    'audio_file': f'{phrase}.wav',
                    'augmentation': aug,
                    'phrase': phrase
                }
                
                # Add MFCC features
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = np.random.uniform(-500, 500)
                    features[f'mfcc_{i}_std'] = np.random.uniform(50, 200)
                
                # Add spectral features
                features.update({
                    'spectral_centroid_mean': np.random.uniform(1000, 5000),
                    'spectral_centroid_std': np.random.uniform(100, 500),
                    'spectral_rolloff_mean': np.random.uniform(2000, 8000),
                    'spectral_rolloff_std': np.random.uniform(200, 800),
                    'zcr_mean': np.random.uniform(0.01, 0.1),
                    'zcr_std': np.random.uniform(0.001, 0.01),
                    'rms_mean': np.random.uniform(0.01, 0.1),
                    'rms_std': np.random.uniform(0.001, 0.01)
                })
                
                # Add chroma features
                for i in range(12):
                    features[f'chroma_{i}_mean'] = np.random.uniform(0, 1)
                
                audio_features.append(features)
    
    audio_df = pd.DataFrame(audio_features)
    audio_df.to_csv('data/processed/audio_features.csv', index=False)
    print("✅ Dummy audio features created!")

def process_real_images():
    """Process real images if they exist"""
    images_dir = 'data/external/images/'
    if os.path.exists(images_dir) and any(os.listdir(images_dir)):
        print("\n🖼️  Processing real images...")
        try:
            from src.image_processing import ImageProcessor
            image_processor = ImageProcessor()
            image_features = image_processor.process_all_images(images_dir, 'data/processed/image_features.csv')
            print("✅ Real image features extracted!")
        except Exception as e:
            print(f"❌ Error processing images: {e}")
            print("Using dummy image features instead...")
            create_dummy_features()
    else:
        print("\n📷 No real images found, using dummy features...")
        create_dummy_features()

def process_real_audio():
    """Process real audio if it exists"""
    audio_dir = 'data/external/audio/'
    if os.path.exists(audio_dir) and any(os.listdir(audio_dir)):
        print("\n🎵 Processing real audio...")
        try:
            from src.audio_processing import AudioProcessor
            audio_processor = AudioProcessor()
            audio_features = audio_processor.process_all_audio(audio_dir, 'data/processed/audio_features.csv')
            if not audio_features.empty:
                print("✅ Real audio features extracted!")
                return  # Success, don't create dummy data
            else:
                print("❌ Audio processing returned empty features")
                raise Exception("Empty audio features")
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            print("Using dummy audio features instead...")
            create_dummy_audio_features()
    else:
        print("\n🎵 No real audio found, using dummy features...")
        create_dummy_audio_features()


def create_dummy_image_features():
    """Create dummy image features only"""
    print("Creating dummy image features...")
    
    # Create dummy image features
    image_features = []
    members = ['member1', 'member2', 'member3']
    expressions = ['neutral', 'smiling', 'surprised']
    augmentations = ['original', 'rotated_15', 'flipped', 'grayscale', 'brightened']
    
    for member in members:
        for expr in expressions:
            for aug in augmentations:
                features = {
                    'member_id': member,
                    'image_file': f'{expr}.jpg',
                    'augmentation': aug,
                    'expression': expr,
                    'hist_r_mean': np.random.uniform(50, 150),
                    'hist_r_std': np.random.uniform(10, 30),
                    'hist_g_mean': np.random.uniform(50, 150),
                    'hist_g_std': np.random.uniform(10, 30),
                    'hist_b_mean': np.random.uniform(50, 150),
                    'hist_b_std': np.random.uniform(10, 30),
                    'hog_mean': np.random.uniform(100, 200),
                    'hog_std': np.random.uniform(20, 40),
                    'edge_density': np.random.uniform(0.1, 0.5),
                    'texture_mean': np.random.uniform(100, 200),
                    'texture_std': np.random.uniform(20, 40)
                }
                image_features.append(features)
    
    image_df = pd.DataFrame(image_features)
    image_df.to_csv('data/processed/image_features.csv', index=False)
    print("✅ Dummy image features created!")

def create_dummy_audio_features():
    """Create dummy audio features only"""
    print("Creating dummy audio features...")
    
    # Create dummy audio features
    audio_features = []
    members = ['member1', 'member2', 'member3']
    phrases = ['yes_approve', 'confirm_transaction']
    augmentations = ['original', 'pitch_shifted', 'time_stretched']
    
    for member in members:
        for phrase in phrases:
            for aug in augmentations:
                features = {
                    'member_id': member,
                    'audio_file': f'{phrase}.wav',
                    'augmentation': aug,
                    'phrase': phrase
                }
                
                # Add MFCC features
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = np.random.uniform(-500, 500)
                    features[f'mfcc_{i}_std'] = np.random.uniform(50, 200)
                
                # Add spectral features
                features.update({
                    'spectral_centroid_mean': np.random.uniform(1000, 5000),
                    'spectral_centroid_std': np.random.uniform(100, 500),
                    'spectral_rolloff_mean': np.random.uniform(2000, 8000),
                    'spectral_rolloff_std': np.random.uniform(200, 800),
                    'zcr_mean': np.random.uniform(0.01, 0.1),
                    'zcr_std': np.random.uniform(0.001, 0.01),
                    'rms_mean': np.random.uniform(0.01, 0.1),
                    'rms_std': np.random.uniform(0.001, 0.01)
                })
                
                # Add chroma features
                for i in range(12):
                    features[f'chroma_{i}_mean'] = np.random.uniform(0, 1)
                
                audio_features.append(features)
    
    audio_df = pd.DataFrame(audio_features)
    audio_df.to_csv('data/processed/audio_features.csv', index=False)
    print("✅ Dummy audio features created!")

    
def main():
    """Main pipeline execution"""
    # Check dependencies first
    print("🔍 Checking dependencies...")
    check_dependencies()
    
    # Check if data files exist
    print("\n📁 Checking data files...")
    if not check_data_files():
        print("❌ Please add the required CSV files to data/raw/ directory")
        return
    
    # Load and validate existing data
    social_df, transactions_df = load_and_validate_data()
    if social_df is None or transactions_df is None:
        return
    
    # Create directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/external/images', exist_ok=True)
    os.makedirs('data/external/audio', exist_ok=True)
    
    print("\n✅ All dependencies and data files are ready!")
    print("🚀 Starting the product recommendation system pipeline...")
    
    # Step 1: Data Merge and Feature Engineering
    print("\n📊 Step 1: Merging datasets and engineering features...")
    try:
        from src.data_processing import DataProcessor
        data_processor = DataProcessor()
        
        # Merge the existing datasets
        merged_df = data_processor.load_data('data/raw/customer_social_profiles.csv', 
                                           'data/raw/customer_transactions.csv')
        
        # Engineer features
        customer_features = data_processor.engineer_features(merged_df)
        customer_features.to_csv('data/processed/merged_dataset.csv', index=False)
        
        print(f"✅ Data merge completed! Final dataset: {customer_features.shape[0]} rows, {customer_features.shape[1]} columns")
        print("Merged dataset sample:")
        print(customer_features.head(3))
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        print("Creating basic merged dataset as fallback...")
        # Create a simple merged dataset as fallback
        try:
            merged_simple = pd.merge(social_df, transactions_df, 
                                   left_on='customer_id_new', 
                                   right_on='customer_id_legacy', 
                                   how='left')
            merged_simple.to_csv('data/processed/merged_dataset.csv', index=False)
            print("✅ Basic merged dataset created as fallback")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            return
    
    # Step 2: Process Images and Audio
    print("\n🎭 Step 2: Processing images and audio...")
    
    # Process images (real if available, otherwise dummy)
    process_real_images()
    
    # Process audio (real if available, otherwise dummy)  
    process_real_audio()
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Training machine learning models...")
    try:
        from src.model_training import ModelTrainer
        
        # Load the processed data
        image_features = pd.read_csv('data/processed/image_features.csv')
        audio_features = pd.read_csv('data/processed/audio_features.csv')
        merged_data = pd.read_csv('data/processed/merged_dataset.csv')
        
        print(f"Image features: {image_features.shape}")
        print(f"Audio features: {audio_features.shape}")
        print(f"Merged data: {merged_data.shape}")
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Train facial recognition model
        print("\n🧠 Training facial recognition model...")
        X_face, y_face = model_trainer.prepare_facial_data(image_features)
        model_trainer.train_model(X_face, y_face, 'random_forest', 'facial_recognition')
        
        # Train voice verification model  
        print("\n🎙️ Training voice verification model...")
        X_voice, y_voice = model_trainer.prepare_audio_data(audio_features)
        model_trainer.train_model(X_voice, y_voice, 'random_forest', 'voice_verification')
        
        # Train product recommendation model
        print("\n🛍️ Training product recommendation model...")
        X_product, y_product = model_trainer.prepare_product_data(merged_data)
        model_trainer.train_model(X_product, y_product, 'random_forest', 'product_recommendation')
        
        # Save models
        model_trainer.save_models('models')
        
        print("✅ All models trained and saved successfully!")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        print("Models were not trained, but the pipeline completed other steps.")
        return
    
    # Final summary
    print("\n🎉 Pipeline completed successfully!")
    print("\n📋 Next steps to enhance the system:")
    print("1. Add team member images to: data/external/images/member1/, member2/, etc.")
    print("   - Include: neutral.jpg, smiling.jpg, surprised.jpg for each member")
    print("2. Add team member audio to: data/external/audio/member1/, member2/, etc.")
    print("   - Include: yes_approve.wav, confirm_transaction.wav for each member")
    print("3. Run the simulation: python run_simulation.py")
    print("4. Run main.py again to process real image/audio data")

if __name__ == "__main__":
    main()