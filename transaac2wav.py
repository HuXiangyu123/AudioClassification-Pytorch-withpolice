from pydub import AudioSegment
import os

# Set the path to the ffmpeg executable
AudioSegment.converter = r'E:\FF\ffmpeg-2023-12-18-git-be8a4f80b9-full_build\bin\ffmpeg.exe'

# Paths to the AAC and WAV folders
aac_folder_path = r'E:\pypro\AudioClassification-Pytorch-v2-6\dataset\second'
wav_folder_path = r'E:\pypro\AudioClassification-Pytorch-v2-6\dataset\second_wav'

# Ensure the output directory exists
if not os.path.exists(wav_folder_path):
    os.makedirs(wav_folder_path)

# List all AAC files in the folder
aac_files = [f for f in os.listdir(aac_folder_path) if f.endswith('.aac')]

for aac_file in aac_files:
    # Construct the full path to the AAC file
    aac_file_path = os.path.join(aac_folder_path, aac_file)

    try:
        # Read the AAC file
        audio = AudioSegment.from_file(aac_file_path, format='aac')

        # Construct the corresponding WAV file path
        wav_file_path = os.path.join(wav_folder_path, aac_file[:-4] + '.wav')

        # Export the audio data to a WAV file
        audio.export(wav_file_path, format='wav')
        print(f"Converted {aac_file} to WAV.")

    except Exception as e:
        print(f"Error processing {aac_file}: {e}")

print("Conversion process completed.")
