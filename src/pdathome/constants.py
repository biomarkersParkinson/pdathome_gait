import os

from dotenv import load_dotenv
from paradigma.constants import DataColumns

load_dotenv()

# paths
PATH_DATA = os.getenv('PATH_DATA')
PATH_SENSOR_DATA = os.path.join(PATH_DATA, 'sensor_data')
PATH_ANNOTATIONS_PD = os.path.join(PATH_DATA, 'video_annotations', 'pd')
PATH_ANNOTATIONS_CONTROLS = os.path.join(PATH_DATA, 'video_annotations', 'controls')

PATH_INPUT = os.path.join(PATH_DATA, 'preprocessed_data', '0.input')
PATH_GAIT_FEATURES = os.path.join(PATH_DATA, 'preprocessed_data', '1.gait_features')
PATH_GAIT_PREDICTIONS = os.path.join(PATH_DATA, 'preprocessed_data', '2.gait_predictions')
PATH_ARM_ACTIVITY_FEATURES = os.path.join(PATH_DATA, 'preprocessed_data', '3.arm_activity_features')
PATH_ARM_ACTIVITY_PREDICTIONS = os.path.join(PATH_DATA, 'preprocessed_data', '4.arm_activity_predictions')

PATH_CLINICAL_DATA = os.path.join(PATH_DATA, 'preprocessed_data', '0.input', 'clinical_data')
PATH_DATAFRAMES = os.path.join(PATH_INPUT, 'dataframes')
PATH_CLASSIFIERS = os.path.join(PATH_INPUT, 'classifiers')
PATH_SCALERS = os.path.join(PATH_INPUT, 'scalers')
PATH_THRESHOLDS = os.path.join(PATH_INPUT, 'thresholds')

# column names
ID_COLNAME = 'id'
L_ARM_ACTIVITY_ANNOTATION_COLNAMES = ['tier', 'nan', 'start_s', 'end_s', 'duration_s', 'code']

# descriptives
MOST_AFFECTED_SIDE = 'MAS'
LEAST_AFFECTED_SIDE = 'LAS'
RIGHT_WRIST = 'RW'
LEFT_WRIST = 'LW'

# classifiers
LOGISTIC_REGRESSION = 'logreg'
RANDOM_FOREST = 'rf'

# lists of participants
L_PD_IDS = ['hbv' + x for x in ['002', '012', '014', '015', '016', '017',
                                '022', '024', '039', '043', '047',
                                '054', '065', '077', '079', '090', '013',
                                '018', '023', '038', '058', '063'
                                ]
]

L_HC_IDS = ['hbv' + x for x in ['053', '072', '073', '082', '083', '084',
                                '087', '091', '093', '097', '099',
                                '100', '106', '108', '109', '110', '112',
                                '115', '117', '122', '128', '136', '081'
                                ]
]

# several hc only have pre, no post data in the Matlab file
L_PRE_IDS = ['hbv115', 'hbv117', 'hbv122', 'hbv136']

L_TREMOR_IDS = ['hbv' + x for x in ['012', '013', '017', '018', '022',
                                    '023', '038', '090']]

L_W_PARTS = ['hbv013', 'hbv018', 'hbv023', 'hbv038', 'hbv058', 'hbv063', 'hbv080']

# some participants have the sensor rotated 180 degrees
L_L_NORMAL = ['hbv' + x for x in ['002', '012', '013', '014', '015', '017', '018',
                                  '022', '023', '024', '038', '039', '043', '047',
                                  '054', '063', '077', '079', '090', '053', '072',
                                  '073', '081', '082', '083', '084', '087', '091',
                                  '093', '097', '099', '100', '106', '108', '109',
                                  '110', '112', '115', '117', '122', '128', '136']]
L_R_NORMAL = ['hbv' + x for x in ['002', '012', '013', '014', '015', '017', '018',
                                  '022', '023', '024', '038', '039', '043', '047',
                                  '054', '063', '077', '079', '090', '053', '072',
                                  '073', '081', '082', '083', '084', '087', '091',
                                  '093', '097', '099', '100', '106', '108', '109',
                                  '110', '112', '115', '117', '122', '128', '136',
                                  '058']]

# parameters
SAMPLING_FREQUENCY = 200 # Hz
DOWNSAMPLED_FREQUENCY = 100 # Hz


# mapping
# Annotation labels
D_LABELS = {
    'General protocol structure': {
        1.0: 'Start synchronization of sensors',
        2.0: 'Installation of sensors',
        3.0: 'Motor examination (in OFF state)',
        4.0: 'Free living part 1 (in OFF state)',
        5.0: 'Questionnaires',
        6.0: 'Motor examination (in ON state)',
        7.0: 'Free living part 2 (in ON state)',
        8.0: 'Taking off sensors',
        9.0: 'End synchronization of sensors',
    },
    'Mobility states during free living parts and questionnaires': {
        1.0: 'Sitting',
        2.0: 'Standing',
        3.0: 'Walking',
        4.0: 'Turning',
        5.0: 'Stair climbing',
        5.1: 'Walking upstairs',
        5.2: 'Walking downstairs',
        6.0: 'Laying',
        7.1: 'Sit-to-stand (normal chair)',
        7.2: 'Stand-to-sit (normal chair)',
        7.3: 'Sit-to-stand (low chair/couch)',
        7.4: 'Stand-to-sit (low chair/couch)',
        7.5: 'Lie-to-stand',
        7.6: 'Stand-to-lie',
        7.7: 'Sit-to-lie',
        7.8: 'Lie-to-sit',
        8.0: 'Exercising on a crosstrainer',
        9.0: 'Cycling',
        10.0: 'Running',
        11.0: 'Driving a motorized scooter',
        12.0: 'Driving a car',
        13.0: 'Doing push-up exercises',
        99.0: 'Unknown',
    },
    'Clinical tests during motor examination parts': {
        1.1: 'Finger tapping left hand',
        1.2: 'Finger tapping right hand',
        2.1: 'Opening and closing left hand',
        2.2: 'Opening and closing right hand',
        3.1: 'Pronation supination left hand',
        3.2: 'Pronation supination right hand',
        4.1: 'Toe tapping left toe',
        4.2: 'Toe tapping right toe',
        5.1: 'Leg agility left leg',
        5.2: 'Leg agility right leg',
        6.0: 'Arise from chair',
        7.1: 'Walking pattern/TUG clockwise',
        7.2: 'Walking pattern/TUG anti-clockwise',
        8.0: 'Postural stability',
        9.1: 'Posutral tremor left hand',
        9.2: 'Postural tremor right hand',
        10.1: 'Kinetic tremor left hand',
        10.2: 'Kinetic tremor right hand',
    },
    'Medication intake and motor status of the patient: On and OFF': {
        1.0: 'Medication intake',
        2.0: 'ON state',
        3.0: 'OFF state',
    },
    'Tremor arm': {
        99.0: 'Not assessable for more than 3 consecutive seconds',
        98.0: 'No tremor with significant upper limb activity',
        97.0: 'Tremor with significant upper limb activity',
        96.0: 'Periodic activity of hand/arm similar frequency to tremor',
        3.0: 'Severe tremor',
        2.0: 'Moderate tremor',
        1.0: 'Slight or mild tremor',
        0.0: 'No tremor'
    },
    'Arm': {
        1.0: 'Gait without other behaviours or other positions',
        2.0: 'Point at something / waving (raising hand)',
        3.0: 'Making hand gestures other than pointing',
        4.0: 'Holding an object in forward position (this includes the whole behaviour including grabbing the object) (*Sometimes people hold their hands forward when walking)',
        5.0: 'Holding an object in downward position (e.g., a book) (this includes the whole behaviour including grabbing the object)',
        6.0: 'Grabbing an object (other than 7) / putting something down',
        7.0: 'Grabbing phone or similar object from pocket',
        8.0: 'Calling with phone',
        9.0: 'Using the phones touchscreen',
        10.0: 'Opening by grabbing (door, window, fridge, cabinet)',
        11.0: 'Closing by grabbing (door, window, fridge)',
        12.0: 'Closing by throwing, pushing back (door, window, fridge)',
        13.0: 'Fixing clothes',
        14.0: 'Fixing devices',
        15.0: 'Holding hands behind back',
        16.0: 'Hand in front trouser pocket',
        17.0: 'Hand in jacket pocket',
        18.0: 'Washing hand',
        19.0: 'Touching face',
        20.0: 'Patting pet',
        21.0: 'Putting on jacket arms',
        22.0: 'Using hand for support in upward position (e.g. holding wall, door, bike steering wheel) - including grabbing',
        23.0: 'Using hand for support in downward position (e.g. sitting down in chair, kitchen counter, holding bike saddle) - including grabbing',
        24.0: 'Turning on/off lights',
        25.0: 'Putting on jacket shoulders',
        26.0: 'Holding hands in front (lijkt op 4)',
        27.0: 'Opening high (cupboard)',
        28.0: 'Taking off jacket (zipping, pulling off shoulders, etc.)',
        29.0: 'Hand on waist',
        30.0: 'Mowing lawn',
        31.0: 'Picking something up from the floor / object low',
        32.0: 'Hand on chest',
        33.0: 'Rubbing hands / moving hands high frequency in front op body / stirring bowl',
        34.0: 'Closing high (cupboard)',
        35.0: 'Dog leash', 
        36.0: 'Hanging up / picking up object high',
        37.0: 'Hands on pockets',
        38.0: 'Hands folded across',
        39.0: 'Hanging clothes on chair (e.g., jacket)',
        40.0: 'Pulling chair backward',
        41.0: 'Untieing / putting on dog leash',
        42.0: 'Pushing chair forward',
        43.0: 'Carrying large object (e.g., chair)',
        44.0: 'Opening by pushing forward',
        45.0: 'Holding an object behind',
        46.0: 'Vacuum cleaning',
        47.0: 'Brushing teeth',
        48.0: 'Holding an object forward + moving around object',
        49.0: 'Pulling object behind (e.g., trash container)',
        50.0: 'Running / hurrying',
        51.0: 'Taking off jacket - pulling off shoulders',
        52.0: 'Hand on butt',
        53.0: 'Hand clasped in front',
        54.0: 'Walking downstairs',
        55.0: 'Make bed',
        56.0: 'Scratching back',
        57.0: 'Cleaning with cloth (kitchen, table)',
        58.0: 'With walking stick',
        59.0: 'Taking off jacket - arm',
        60.0: 'Assisting other arm / hand (e.g., when grabbing something from pocket)',
        61.0: 'Holding an object to chest, as if cuddling',
        99.0: 'Transition to/from sitting down',
        100.0: 'cant assess',
        101.0: 'cant assess'
    }
}

D_TIERS_MAP = {
    'Mobility states during free living parts and questionnaires': 'free_living',
    'Clinical tests during motor examination parts': 'clinical_tests',
    'Medication intake and motor status of the patient: On and OFF': 'med_and_motor_status',
    'Tremor arm': 'tremor',
    'Left arm': 'left_arm',
    'Right arm': 'right_arm',
}