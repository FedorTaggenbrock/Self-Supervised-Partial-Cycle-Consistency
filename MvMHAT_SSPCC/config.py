# dataset
TRAIN_DATASET = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
TEST_DATASET = ['circleRegion', 'innerShop', 'movingView', 'park', 'playground', 'shopFrontGate', 'shopSecondFloor', 'shopSideGate', 'shopSideSquare', 'southGate']
VIEWS = 3
FRAMES = 2
ROOT_DIR = './DATA/DIVO'

# training
#TRAIN_GPUS = '0'


NETWORK = 'resnet'
# RE_ID = True
# TRAIN_RESUME = "/cross-view/DIVOTrack/Cross_view_Tracking/MvMHAT/models/Standard_MvMHAT_model.pth"  #Only used if RE_ID is True


# parameters
MARGIN = 0.5 #For their cycle loss
DATASET_SHUFFLE = 0  #shuffles the batch contents.
LOADER_SHUFFLE = 1

# inference
INF_ID = 'model'
DISPLAY = 0
INFTY_COST = 1e+5
RENEW_TIME = 30
DETECTION_DIR = '/cross-view/DIVOTrack/datasets/DIVO/images/dets/detection_results/'
TESTBB_DIR = '/cross-view/DIVOTrack/datasets/DIVO/test_bb/'

