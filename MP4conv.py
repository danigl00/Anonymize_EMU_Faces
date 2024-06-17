from moviepy.editor import VideoFileClip


def convert_m2t_to_mp4(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file, audio=False)
    clip.close()

# Replace these with your input and output file paths
input_file = 'seizure.m2t'
output_file = 'seizure.mp4'

convert_m2t_to_mp4(input_file, output_file)