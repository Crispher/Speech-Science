# WAV_FILE_PATH = 'data/audio.wav'
# WAV_FILE_PATH = 'data/short.wav'
WAV_FILE_PATH = 'data/pi.wav'
# SEGMENTED_PATH = 'data/segmented/'
SEGMENTED_PATH = 'data/PI_segmented/'
# MFCC_PATH = 'data/mfcc_feature/'
MFCC_PATH = 'data/PI_feature/'


OUTPUT_PATH = 'data/output_debug/'
NODE_PATH = 'data/output_debug/nodes/'

SILENCE_THRESHOLD = 42 # dB
MINIMUN_SEGMENT_DURATION = 0.5 # sec

N_MFCC = 25
INFINITY = 1E10

R_CONSTANT = 3
L_CONSTANT = 14 # 0.24s
THETA = 0.95
WINDOW_WIDTH = L_CONSTANT / 2

DISTORTION_THRESHOLD = 0.95