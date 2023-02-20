# colors
COLOR_LUMEN = (216/255, 101/255, 79/255)
COLOR_PLAQUE = (241/255, 214/255, 145/255)
COLOR_LUMEN_DARK = (55/255, 22/255, 15/255)
COLOR_PLAQUE_DARK = (81/255, 69/255, 40/255)
COLOR_LEFT = (123/255, 50/255, 148/255)
COLOR_LEFT_LIGHT = (173/255, 100/255, 198/255)
COLOR_LEFT_HEX = "#7b3294;"
COLOR_RIGHT = (0, 136/255, 55/255)
COLOR_RIGHT_LIGHT = (30/255, 196/255, 85/255)
COLOR_RIGHT_HEX = "#008837;"
COLOR_UNSELECTED = (255, 255, 255)
COLOR_SELECTED = (200, 200, 255)

# unicode symbols
SYM_YES = "\u2714"
SYM_NO = "\u2716"
SYM_ENDASH = "\u2013"
SYM_UNSAVED_CHANGES = "\u25CF"

# global execution flags
EXPAND_PATIENTS = True
SHOW_MODEL_MISMATCH_WARNING = False

# global parameter constants
MIN_CLUSTER_SIZE = 20000 # minimal cluster size (voxels) computed by automatic segmentation
INITIAL_NR_MAPS = 10 # initial number of latent space maps to show on the FlowCompModule