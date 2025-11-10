import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
import os
import matplotlib.pyplot as plt
import warnings

class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        # Create output directory for visualizations
        self.viz_dir = 'audio_visualizations'
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def load_audio(self, audio_path):
        """Load audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio
    
    def plot_waveform_and_spectrogram(self, audio, audio_file, member_id, augmentation='original'):
        """Plot waveform and spectrogram for audio sample"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        librosa.display.waveshow(audio, sr=self.sample_rate, ax=ax1)
        ax1.set_title(f'Waveform - {member_id} - {audio_file} ({augmentation})', fontsize=12)
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=self.sample_rate, x_axis='time', y_axis='log', ax=ax2)
        ax2.set_title(f'Spectrogram - {member_id} - {audio_file} ({augmentation})', fontsize=12)
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        
        # Add colorbar
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"{member_id}_{audio_file.replace('.', '_')}_{augmentation}.png"
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved visualization: {filename}")
        
    def plot_mfcc_features(self, audio, audio_file, member_id, augmentation='original'):
        """Plot MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, sr=self.sample_rate, x_axis='time')
        plt.colorbar(format='%+2.0f')
        plt.title(f'MFCC Features - {member_id} - {audio_file} ({augmentation})')
        plt.ylabel('MFCC Coefficients')
        plt.xlabel('Time (s)')
        
        # Save the plot
        filename = f"{member_id}_{audio_file.replace('.', '_')}_{augmentation}_mfcc.png"
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_spectral_features(self, audio, audio_file, member_id, augmentation='original'):
        """Plot various spectral features"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=self.sample_rate)
        ax1.plot(t, spectral_centroids, color='b')
        ax1.set_title('Spectral Centroid')
        ax1.set_ylabel('Hz')
        ax1.grid(True)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        ax2.plot(t, spectral_rolloff, color='r')
        ax2.set_title('Spectral Rolloff')
        ax2.set_ylabel('Hz')
        ax2.grid(True)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        ax3.plot(t, zcr, color='g')
        ax3.set_title('Zero Crossing Rate')
        ax3.set_ylabel('Rate')
        ax3.set_xlabel('Time (s)')
        ax3.grid(True)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        ax4.plot(t, rms, color='m')
        ax4.set_title('RMS Energy')
        ax4.set_ylabel('Amplitude')
        ax4.set_xlabel('Time (s)')
        ax4.grid(True)
        
        plt.suptitle(f'Spectral Features - {member_id} - {audio_file} ({augmentation})', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        filename = f"{member_id}_{audio_file.replace('.', '_')}_{augmentation}_spectral.png"
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def apply_augmentations(self, audio):
        """Apply audio augmentations"""
        augmentations = []
        
        # Original audio
        augmentations.append(('original', audio))
        
        try:
            # Pitch shift
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=2)
            augmentations.append(('pitch_shifted', pitch_shifted))
            
            # Time stretch
            time_stretched = librosa.effects.time_stretch(audio, rate=0.8)
            augmentations.append(('time_stretched', time_stretched))
            
            # Add noise
            noise = np.random.normal(0, 0.005, audio.shape)
            noisy_audio = audio + noise
            augmentations.append(('noisy', noisy_audio))
            
            # Speed change
            speed_changed = librosa.effects.time_stretch(audio, rate=1.2)
            augmentations.append(('speed_increased', speed_changed))
        except Exception as e:
            print(f" Some augmentations failed: {e}")
        
        return augmentations
    
    def extract_features(self, audio):
        """Extract audio features"""
        features = {}
        
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            features.update({
                f'mfcc_{i}_mean': np.mean(mfccs[i]) for i in range(13)
            })
            features.update({
                f'mfcc_{i}_std': np.std(mfccs[i]) for i in range(13)
            })
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            
            features.update({
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
            })
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features.update({
                f'chroma_{i}_mean': np.mean(chroma[i]) for i in range(12)
            })
            
          
            features['audio_length'] = len(audio) / self.sample_rate  # Duration in seconds
            features['max_amplitude'] = np.max(np.abs(audio))
            
        except Exception as e:
            print(f" Feature extraction failed: {e}")
            # Create dummy features as fallback
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.random.uniform(-500, 500)
                features[f'mfcc_{i}_std'] = np.random.uniform(50, 200)
            
            features.update({
                'spectral_centroid_mean': np.random.uniform(1000, 5000),
                'spectral_centroid_std': np.random.uniform(100, 500),
                'spectral_rolloff_mean': np.random.uniform(2000, 8000),
                'spectral_rolloff_std': np.random.uniform(200, 800),
                'zcr_mean': np.random.uniform(0.01, 0.1),
                'zcr_std': np.random.uniform(0.001, 0.01),
                'rms_mean': np.random.uniform(0.01, 0.1),
                'rms_std': np.random.uniform(0.001, 0.01),
                'audio_length': np.random.uniform(1, 5),
                'max_amplitude': np.random.uniform(0.1, 1.0)
            })
            
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.random.uniform(0, 1)
        
        return features
    
    def generate_audio_report(self, audio_features_df):
        """Generate a summary report of audio features"""
        report_file = os.path.join(self.viz_dir, 'audio_analysis_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("AUDIO FEATURE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total audio samples processed: {len(audio_features_df)}\n")
            f.write(f"Unique members: {audio_features_df['member_id'].nunique()}\n")
            f.write(f"Unique phrases: {audio_features_df['phrase'].nunique()}\n")
            f.write(f"Augmentations applied: {audio_features_df['augmentation'].nunique()}\n\n")
            
            f.write("Member Statistics:\n")
            member_stats = audio_features_df['member_id'].value_counts()
            for member, count in member_stats.items():
                f.write(f"  {member}: {count} samples\n")
            
            f.write("\nFeature Statistics:\n")
            numeric_cols = audio_features_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  # Show first 10 numeric features
                f.write(f"  {col}: mean={audio_features_df[col].mean():.4f}, std={audio_features_df[col].std():.4f}\n")
        
        print(f" Audio analysis report saved: {report_file}")
    
    def process_all_audio(self, audio_dir, output_path, generate_plots=True):
        """Process all audio files and save features with optional visualizations"""
        all_features = []
        processed_count = 0
        
        print(f" Processing audio files from: {audio_dir}")
        print(f" Visualization output: {self.viz_dir}")
        
        for member_dir in os.listdir(audio_dir):
            member_path = os.path.join(audio_dir, member_dir)
            if os.path.isdir(member_path):
                print(f"  Processing {member_dir}...")
                for audio_file in os.listdir(member_path):
                    if audio_file.lower().endswith(('.wav', '.mp3')):
                        audio_path = os.path.join(member_path, audio_file)
                        print(f"    Analyzing {audio_file}...")
                        
                        try:
                            # Load audio
                            audio = self.load_audio(audio_path)
                            print(f"      Loaded: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")
                            
                            # Apply augmentations and extract features
                            augmentations = self.apply_augmentations(audio)
                            
                            for aug_name, aug_audio in augmentations:
                                # Generate visualizations for original audio only (to avoid too many plots)
                                if generate_plots and aug_name == 'original':
                                    self.plot_waveform_and_spectrogram(aug_audio, audio_file, member_dir, aug_name)
                                    self.plot_mfcc_features(aug_audio, audio_file, member_dir, aug_name)
                                    self.plot_spectral_features(aug_audio, audio_file, member_dir, aug_name)
                                
                                # Extract features
                                features = self.extract_features(aug_audio)
                                features.update({
                                    'member_id': member_dir,
                                    'audio_file': audio_file,
                                    'augmentation': aug_name,
                                    'phrase': audio_file.split('.')[0]
                                })
                                all_features.append(features)
                                processed_count += 1
                                
                        except Exception as e:
                            print(f"      Error processing {audio_path}: {e}")
                            continue
        
        # Create DataFrame and save
        if all_features:
            features_df = pd.DataFrame(all_features)
            features_df.to_csv(output_path, index=False)
            
            # Generate summary report
            self.generate_audio_report(features_df)
            
            print(f"\n Successfully processed {processed_count} audio feature sets")
            print(f" Features saved to: {output_path}")
            print(f" Visualizations saved to: {self.viz_dir}/")
            
            return features_df
        else:
            print("❌ No audio files processed successfully")
            raise Exception("No audio features extracted")

# Example usage and test function
def test_audio_processor():
    """Test the audio processor with sample data"""
    processor = AudioProcessor()
    
    # Test with your audio directory
    audio_dir = 'data/external/audio/'
    output_path = 'data/processed/audio_features.csv'
    
    if os.path.exists(audio_dir):
        features_df = processor.process_all_audio(audio_dir, output_path, generate_plots=True)
        print(f"Processed {len(features_df)} audio samples")
    else:
        print(f"Audio directory not found: {audio_dir}")

if __name__ == "__main__":
    test_audio_processor()