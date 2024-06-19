from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cut_video(input_file, output_file, start_time, end_time):
    ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_file)

# Example usage
input_file = "p302_seizure_anonymized.mp4"
output_file = "Edit_p302_seizure_anonymized.mp4"
start_time = 4  # Start time in seconds
end_time = 70  # End time in seconds

cut_video(input_file, output_file, start_time, end_time)