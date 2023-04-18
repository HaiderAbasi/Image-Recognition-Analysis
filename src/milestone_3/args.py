# Fastest?
run_fastest = False
# Demo type, eg. image, video, or webcam
demo = "image"

# Experiment name
experiment_name = None

if run_fastest:
    name = "yolox-nano"
else:
    name = "yolox-s"

# Path to images or video
path = "data/raw/test/103274711606405.jpeg"

# Webcam demo camera id
camid = 0

# Experiment description file
exp_file = None


# Ckpt for eval
if run_fastest:
    ckpt = "data/models/yolox_nano.pth"
else:    
    ckpt = "data/models/yolox_s.pth"

# Device to run the model, can either be cpu or gpu
device = "cpu"

# Test conf threshold
conf = 0.3

# Test nms threshold
nms = 0.3

# Test img size
tsize = 640

# Adopting mix precision evaluating
fp16 = False

# To be compatible with older versions
legacy = False
