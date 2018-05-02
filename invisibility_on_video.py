"""
Written by Michael Nation 5/2/2018
"""

from InvisibilityEngine import *

def run_invisibility_on_video():
    video_filepath = 'videos/input_video_01.mp4'
    video = cv2.VideoCapture(video_filepath)

    grabbed, frame = video.read()
    if not grabbed:
        print('Failed to read the input video.')
        return

    # The first frame of the video will be used as the background image.
    # There should not be a person in the first frame
    img_background = frame

    # input_video_01.mp4 was recorderd with 15 fps.
    # To reduce processing time, only 5 fps will be used. i.e. 1 of every 3 images are used.
    fps = 5
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('videos/invisibility_demo_01.mp4', fourcc, fps, (width*2, height))

    inv_engine = InvisibilityEngine()

    frame_num = 0
    while grabbed:
        img_processed = inv_engine.run_invisibility_on_image(frame, img_background)

        combined_img = np.empty((height, width * 2, 3)).astype(np.uint8)
        combined_img[:, :width] = frame
        combined_img[:, width:] = img_processed
        video_out.write(combined_img)

        grabbed, frame = get_next_frame(video)
        frame_num += 1

        if frame_num % 5 == 0:
            print('frame num', frame_num)

    video.release()
    video_out.release()

def get_next_frame(video):
    """Skips the first two images. Returns the third."""
    video.read()
    video.read()

    return video.read()


if __name__== '__main__':
    run_invisibility_on_video()