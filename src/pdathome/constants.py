import os

from dataclasses import dataclass, field
from dotenv import load_dotenv
from matplotlib import colors, colormaps
from typing import Any, Dict, List

from paradigma.constants import DataColumns

# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class Paths:
    PATH_DATA_FOLDER: str
    PATH_INPUT: str
    PATH_SENSOR_DATA: str
    PATH_ANNOTATIONS: str
    PATH_CLINICAL_DATA: str
    PATH_OUTPUT: str
    PATH_CLASSIFIERS: str
    PATH_SCALERS: str
    PATH_THRESHOLDS: str
    PATH_COEFFICIENTS: str
    PATH_PREPROCESSED_DATA: str
    PATH_PREPARED_DATA: str
    PATH_GAIT_FEATURES: str
    PATH_GAIT_PREDICTIONS: str
    PATH_ARM_ACTIVITY_FEATURES: str
    PATH_ARM_ACTIVITY_PREDICTIONS: str

    @classmethod
    def from_env(cls):
        PATH_DATA_FOLDER = os.getenv('PATH_DATA_FOLDER')
        PATH_INPUT = os.getenv('PATH_INPUT_DATA')
        PATH_PREPROCESSED_DATA = os.getenv('PATH_PREPROCESSED_DATA')
        PATH_OUTPUT = os.getenv('PATH_OUTPUT_DATA')
        return cls(
            PATH_DATA_FOLDER = PATH_DATA_FOLDER,
            PATH_INPUT = PATH_INPUT,
            PATH_SENSOR_DATA = os.path.join(PATH_INPUT, 'sensor_data'),
            PATH_ANNOTATIONS = os.path.join(PATH_INPUT, 'video_annotations'),
            PATH_CLINICAL_DATA = os.path.join(PATH_INPUT, 'clinical_data'),
            PATH_OUTPUT = PATH_OUTPUT,
            PATH_CLASSIFIERS = os.path.join(PATH_OUTPUT, 'classifiers'),
            PATH_SCALERS = os.path.join(PATH_OUTPUT, 'scalers'),
            PATH_THRESHOLDS = os.path.join(PATH_OUTPUT, 'thresholds'),
            PATH_COEFFICIENTS = os.path.join(PATH_OUTPUT, 'feature_coefficients'),
            PATH_PREPROCESSED_DATA = PATH_PREPROCESSED_DATA,
            PATH_PREPARED_DATA = os.path.join(PATH_PREPROCESSED_DATA, '0.prepared_data'),
            PATH_GAIT_FEATURES = os.path.join(PATH_PREPROCESSED_DATA, '1.gait_features'),
            PATH_GAIT_PREDICTIONS = os.path.join(PATH_PREPROCESSED_DATA, '2.gait_predictions'),
            PATH_ARM_ACTIVITY_FEATURES = os.path.join(PATH_PREPROCESSED_DATA, '3.arm_activity_features'),
            PATH_ARM_ACTIVITY_PREDICTIONS = os.path.join(PATH_PREPROCESSED_DATA, '4.arm_activity_predictions'),
        )

@dataclass(frozen=True)
class Columns:
    ID: str = 'id'
    TIME: str = DataColumns.TIME
    PRE_OR_POST: str = 'pre_or_post'
    ACCELEROMETER_X: str = DataColumns.ACCELEROMETER_X
    ACCELEROMETER_Y: str = DataColumns.ACCELEROMETER_Y
    ACCELEROMETER_Z: str = DataColumns.ACCELEROMETER_Z
    GRAV_ACCELEROMETER_X: str = f'grav_{DataColumns.ACCELEROMETER_X}'
    GRAV_ACCELEROMETER_Y: str = f'grav_{DataColumns.ACCELEROMETER_Y}'
    GRAV_ACCELEROMETER_Z: str = f'grav_{DataColumns.ACCELEROMETER_Z}'
    GYROSCOPE_X: str = DataColumns.GYROSCOPE_X
    GYROSCOPE_Y: str = DataColumns.GYROSCOPE_Y
    GYROSCOPE_Z: str = DataColumns.GYROSCOPE_Z
    SIDE: str = 'side'
    FREE_LIVING_LABEL: str = 'free_living_label'
    ARM_LABEL: str = 'arm_label'
    TREMOR_LABEL: str = 'tremor_label'
    ACTIVITY_LABEL_MAJORITY_VOTING: str = 'activity_majority_voting'
    GAIT_MAJORITY_VOTING: str = 'gait_majority_voting'
    ARM_LABEL_MAJORITY_VOTING: str = 'arm_activity_majority_voting'
    OTHER_ARM_ACTIVITY_MAJORITY_VOTING: str = 'other_arm_activity_majority_voting'
    PRED_GAIT: str = 'pred_gait'
    PRED_GAIT_PROBA: str = 'pred_gait_proba'
    PRED_OTHER_ARM_ACTIVITY: str = 'pred_other_arm_activity'
    PRED_OTHER_ARM_ACTIVITY_PROBA: str = 'pred_other_arm_activity_proba'
    ANGLE: str = 'angle'
    ANGLE_SMOOTH: str = 'angle_smooth'
    VELOCITY: str = 'velocity'
    WINDOW_NR: str = 'window_nr'
    SEGMENT_NR: str = 'segment_nr'
    SEGMENT_CAT: str = 'segment_cat'
    TRUE_SEGMENT_NR: str = 'true_segment_nr'
    PRED_SEGMENT_NR: str = 'pred_segment_nr'
    TRUE_SEGMENT_CAT: str = 'true_segment_cat'
    PRED_SEGMENT_CAT: str = 'pred_segment_cat'
    L_ACCELEROMETER: List[str] = field(default_factory=lambda: [
        DataColumns.ACCELEROMETER_X, DataColumns.ACCELEROMETER_Y, DataColumns.ACCELEROMETER_Z
    ])
    L_GYROSCOPE: List[str] = field(default_factory=lambda: [
        DataColumns.GYROSCOPE_X, DataColumns.GYROSCOPE_Y, DataColumns.GYROSCOPE_Z
    ])
    L_ARM_ACTIVITY_ANNOTATIONS: List[str] = field(default_factory=lambda: [
        'tier', 'nan', 'start_s', 'end_s', 'duration_s', 'code'
    ])

@dataclass(frozen=True)
class Descriptives:
    MOST_AFFECTED_SIDE: str = 'MAS'
    LEAST_AFFECTED_SIDE: str = 'LAS'
    RIGHT_WRIST: str = 'RW'
    LEFT_WRIST: str = 'LW'
    PRE_MED: str = 'pre'
    POST_MED: str = 'post'
    PARKINSONS_DISEASE: str = 'PD'
    CONTROLS: str = 'HC'

@dataclass(frozen=False)
class Classifiers:
    LOGISTIC_REGRESSION: str = 'logreg'
    RANDOM_FOREST: str = 'rf'
    GAIT_DETECTION_CLASSIFIER_SELECTED: str = 'rf'
    ARM_ACTIVITY_CLASSIFIER_SELECTED: str = 'logreg'

    LOGISTIC_REGRESSION_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda: {
        'penalty': ['l1'],
        'solver': ['saga'],
        'tol': [1e-4, 1e-5],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        'random_state': [22],
    })

    RANDOM_FOREST_PARAM_GRID: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt'],
        'min_samples_split': [20, 50, 100],
        'max_depth': [None, 10, 20],
        'criterion': ['gini'],
        'bootstrap': [True],
        'oob_score': [True],
        'random_state': [22],
    })

    LOGISTIC_REGRESSION_HYPERPARAMETERS: Dict[str, Any] = field(default_factory=lambda: {
        'penalty': 'l1',
        'solver': 'saga',
        'tol': 1e-4,
        'C': 1e-2,
        'random_state': 22,
    })

    RANDOM_FOREST_HYPERPARAMETERS: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_features': 'sqrt',
        'min_samples_split': 25,
        'max_depth': 15,
        'criterion': 'gini',
        'bootstrap': True,
        'oob_score': True,
        'random_state': 22,
    })

@dataclass(frozen=True)
class ParticipantIDs:
    # Parkinson's disease (PD) IDs
    L_PD_IDS: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '002', '012', '014', '015', '016', '017', '022', '024',
        '039', '043', '047', '054', '065', '077', '079', '090',
        '013', '018', '023', '038', '058', '063'
    ]])
    
    # Control IDs
    L_HC_IDS: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '053', '072', '073', '082', '083', '084', '087', '091',
        '093', '097', '099', '100', '106', '108', '109', '110',
        '112', '115', '117', '122', '128', '136', '081'
    ]])

    # PD IDs pre-medication
    L_PRE_IDS: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        'hbv115', 'hbv117', 'hbv122', 'hbv136'
    ]])

    # PD IDs with tremor
    L_TREMOR_IDS: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '012', '013', '017', '018', '022', '023', '038', '090'
    ]])

    # PD IDs with multiple raw files
    L_W_PARTS: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        'hbv013', 'hbv018', 'hbv023', 'hbv038', 'hbv058',
        'hbv063', 'hbv080'
    ]])

    # PD IDs with sensor on left wrist in normal position
    L_L_NORMAL: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '002', '012', '013', '014', '015', '017', '018', '022',
        '023', '024', '038', '039', '043', '047', '054', '063',
        '077', '079', '090', '053', '072', '073', '081', '082',
        '083', '084', '087', '091', '093', '097', '099', '100',
        '106', '108', '109', '110', '112', '115', '117', '122',
        '128', '136'
    ]])

    # PD IDs with sensor on right wrist in normal position
    L_R_NORMAL: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '002', '012', '013', '014', '015', '017', '018', '022',
        '023', '024', '038', '039', '043', '047', '054', '063',
        '077', '079', '090', '053', '072', '073', '081', '082',
        '083', '084', '087', '091', '093', '097', '099', '100',
        '106', '108', '109', '110', '112', '115', '117', '122',
        '128', '136', '058'
    ]])

    # PD IDs with most affected side left
    L_PD_MOST_AFFECTED_LEFT: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '013', '014', '015', '016', '017', '022', '024', '039',
        '043', '047', '077'
    ]])

    # PD IDs with most affected side right
    L_PD_MOST_AFFECTED_RIGHT: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '002', '012', '018', '023', '038', '054', '058', '063',
        '065', '074', '079', '090'
    ]])

    # PD IDs with dominant side left
    L_PD_DOMINANT_LEFT: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '002', '016', '024', '043', '065'
    ]])

    # PD IDs with dominant side right
    L_PD_DOMINANT_RIGHT: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '012', '014', '015', '017', '022', '039', '047', '054',
        '077', '090', '018', '023', '038', '058', '063'
    ]])

    # PD IDs with dominant side both
    L_PD_DOMINANT_BOTH: List[str] = field(default_factory=lambda: ['hbv' + x for x in [
        '013', '079'
    ]])

@dataclass(frozen=True)
class Parameters:
    SAMPLING_FREQUENCY: int = 200
    DOWNSAMPLED_FREQUENCY: int = 100
    SEGMENT_GAP_GAIT: float = 1.5
    SEGMENT_GAP_ARM_ACTIVITY: float = 1.5

@dataclass(frozen=True)
class PlotParameters:
    COLOR_PALETTE: str = 'Paired'
    COLOR_PALETTE_FIRST_COLOR: str = colors.rgb2hex(
        colormaps[(COLOR_PALETTE)](0)
    )
    COLOR_PALETTE_SECOND_COLOR: str = colors.rgb2hex(
        colormaps[(COLOR_PALETTE)](1)
    )

@dataclass(frozen=True)
class GlobalConstants:
    paths: Paths
    columns: Columns
    descriptives: Descriptives
    classifiers: Classifiers
    participant_ids: ParticipantIDs
    parameters: Parameters
    plot_parameters: PlotParameters

global_constants = GlobalConstants(
    paths=Paths.from_env(),
    columns=Columns(),
    descriptives=Descriptives(),  
    classifiers=Classifiers(),    
    participant_ids=ParticipantIDs(), 
    parameters=Parameters(),  
    plot_parameters=PlotParameters() 
)

metric_map = {
    'sens': 'Sensitivity',
    'spec': 'Specificity',
    'acc': 'Accuracy',
    'bacc': 'Balanced accuracy',
}

arm_swing_parameter_map = {
    'range_of_motion_median': 'Median range of motion [deg]',
    'median_rom': 'Median range of motion [deg]',
    'range_of_motion_quantile_95': '95th percentile range of motion [deg]',
    '95p_rom': '95th percentile range of motion [deg]',
    'peak_velocity_median': 'Median peak velocity [deg/s]',
    'peak_velocity_quantile_95': '95th percentile peak velocity [deg/s]',
}

activity_map = {
    'Lie-to-sit': 'Transitioning',
    'Lie-to-stand': 'Transitioning',
    'Sit-to-lie': 'Transitioning',
    'Sit-to-stand (low chair/couch)': 'Transitioning',
    'Sit-to-stand (normal chair)': 'Transitioning',
    'Stand-to-lie': 'Transitioning',
    'Stand-to-sit (low chair/couch)': 'Transitioning',
    'Stand-to-sit (normal chair)': 'Transitioning',
    'Walking downstairs': 'Walking the stairs',
    'Walking upstairs': 'Walking the stairs',
}

segment_map = {
    -1: 'non_gait',
    1: 'short',
    2: 'moderately_long',
    3: 'long',
    4: 'very_long'
}

segment_rename = {
    'short': 'Short [< 5s]',
    'moderately_long': 'Moderately long [5-10s]',
    'long': 'Long [10-20s]',
    'very_long': 'Very long [> 20s]'
}

tiers_labels_map = {
    'General protocol structure' : {
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
    'Mobility states during free living parts and questionnaires' : {
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
    'Clinical tests during motor examination parts' : {
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
        9.1: 'Postural tremor left hand',
        9.2: 'Postural tremor right hand',
        10.1: 'Kinetic tremor left hand',
        10.2: 'Kinetic tremor right hand',
    },
    'Medication intake and motor status of the patient: On and OFF' : {
        1.0: 'Medication intake',
        2.0: 'ON state',
        3.0: 'OFF state',
    },
    'Tremor arm' : {
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
        4.0: 'Holding an object in forward position (including grabbing the object)',
        5.0: 'Holding an object in downward position (e.g., a book, including grabbing the object)',
        6.0: 'Grabbing an object (other than 7) / putting something down',
        7.0: 'Grabbing phone or similar object from pocket',
        8.0: 'Calling with phone',
        9.0: 'Using the phone\'s touchscreen',
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
        26.0: 'Holding hands in front (similar to 4)',
        27.0: 'Opening high (cupboard)',
        28.0: 'Taking off jacket (zipping, pulling off shoulders, etc.)',
        29.0: 'Hand on waist',
        30.0: 'Mowing lawn',
        31.0: 'Picking something up from the floor / object low',
        32.0: 'Hand on chest',
        33.0: 'Rubbing hands / moving hands high frequency in front of body / stirring bowl',
        34.0: 'Closing high (cupboard)',
        35.0: 'Dog leash',
        36.0: 'Hanging up / picking up object high',
        37.0: 'Hands on pockets',
        38.0: 'Hands folded across',
        39.0: 'Hanging clothes on chair (e.g., jacket)',
        40.0: 'Pulling chair backward',
        41.0: 'Untying / putting on dog leash',
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
        55.0: 'Making bed',
        56.0: 'Scratching back',
        57.0: 'Cleaning with cloth (kitchen, table)',
        58.0: 'With walking stick',
        59.0: 'Taking off jacket - arm',
        60.0: 'Assisting other arm / hand (e.g., when grabbing something from pocket)',
        61.0: 'Holding an object to chest, as if cuddling',
        99.0: 'Transition to/from sitting down',
        100.0: 'Cannot assess',
        101.0: 'Cannot assess'
    }
}

tiers_rename = {
    'Mobility states during free living parts and questionnaires' : 'free_living',
    'Clinical tests during motor examination parts': 'clinical_tests',
    'Medication intake and motor status of the patient: On and OFF' : 'med_and_motor_status',
    'Tremor arm' : 'tremor',
    'Left arm' : 'left_arm',
    'Right arm' : 'right_arm'
}

arm_labels_rename = {
    'Gait without other behaviours or other positions': 'Gait without other arm movements',
    'Grabbing an object (other than 7) / putting something down': 'Grabbing',
    'Closing by grabbing (door, window, fridge)': 'Closing by pulling',
    'Closing by throwing, pushing back (door, window, fridge)': 'Closing by pushing',
    'Hanging up / picking up object high': 'Grabbing high',
    'Holding an object in forward position (including grabbing the object)': 'Holding forward',
    'Holding an object in downward position (e.g., a book, including grabbing the object)': 'Holding downward',
    'Holding hands in front (similar to 4)': 'Hands forward',
    'Making hand gestures other than pointing': 'Making hand gestures',
    'Opening by grabbing (door, window, fridge, cabinet)': 'Opening by pulling',
    'Point at something / waving (raising hand)': 'Pointing',
    'Using hand for support in downward position (e.g. sitting down in chair, kitchen counter, holding bike saddle) - including grabbing': 'Using hand for support',
    'Using hand for support in upward position (e.g. holding wall, door, bike steering wheel) - including grabbing': 'Using hand for support',
    'Fixing clothes': 'Fixing clothes',
    'Fixing devices': 'Fixing devices',
    'cant assess': 'Cant assess',
    'Touching face': 'Touching face',
    'Transition to/from sitting down': 'Transition',
    'Grabbing phone or similar object from pocket': 'Grabbing from pocket',
    'Calling with phone': 'Calling with phone',
    'Using the phones touchscreen': 'Using phone touchscreen',
    'Holding hands behind back': 'Hand behind back',
    'Hand in front trouser pocket': 'Hand in front trouser pocket',
    'Hand in jacket pocket': 'Hand in jacket pocket',
    'Washing hand': 'Washing hand',
    'Patting pet': 'Patting pet',
    'Putting on jacket arms': 'Putting on jacket',
    'Turning on/off lights': 'Turning on/off lights',
    'Putting on jacket shoulders': 'Putting on jacket',
    'Opening high (cupboard)': 'Opening by pulling',
    'Taking off jacket (zipping, pulling off shoulders, etc.)': 'Taking off jacket',
    'Hand on waist': 'Hand on waist',
    'Mowing lawn': 'Mowing lawn',
    'Picking something up from the floor / object low': 'Grabbing',
    'Hand on chest': 'Hand on chest',
    'Rubbing hands / moving hands high frequency in front of body / stirring bowl': 'Holding forward and rotating',
    'Closing high (cupboard)': 'Closing by pushing',
    'Dog leash': 'Holding dog leash',
    'Hanging up / picking up object high': 'Grabbing',
    'Hands on pockets': 'Hand on trouser pocket',
    'Hands folded across': 'Hands folded across',
    'Hanging clothes on chair (e.g., jacket)': 'Grabbing',
    'Pulling chair backward': 'Pulling backward',
    'Untying / putting on dog leash': 'Putting on dog leash',
    'Pushing chair forward': 'Pushing forward',
    'Carrying large object (e.g., chair)': 'Holding forward',
    'Opening by pushing forward': 'Opening by pushing',
    'Holding an object behind': 'Hand behind back',
    'Vacuum cleaning': 'Vacuum cleaning',
    'Brushing teeth': 'Brushing teeth',
    'Holding an object forward + moving around object': 'Holding forward and rotating',
    'Pulling object behind (e.g., trash container)': 'Pulling behind back',
    'Running / hurrying': 'Running',
    'Taking off jacket - pulling off shoulders': 'Taking off jacket',
    'Hand on butt': 'Hand in back pocket',
    'Hand clasped in front': 'Hand clasped in front',
    'Walking downstairs': 'Walking stairs',
    'Make bed': 'Making bed',
    'Scratching back': 'Scratching back',
    'Cleaning with cloth (kitchen, table)': 'Cleaning with cloth',
    'With walking stick': 'Using walking stick',
    'Taking off jacket - arm': 'Taking off jacket',
    'Assisting other arm / hand (e.g., when grabbing something from pocket)': 'Holding forward',
    'Holding an object to chest, as if cuddling': 'Hand on chest',
    'Transition to/from sitting down': 'Transition',
    'non_gait': 'Not gait',
    'Cannot assess': 'Cannot assess'
}

updrs_3_map = {
    'right': {
        'hypokinesia': {
            'UPDRS_3_3b': 'Rigidity RUE',
            'UPDRS_3_3d': 'Rigidity RLE',
            'UPDRS_3_4a': 'FT RH',
            'UPDRS_3_5a': 'Movement RH',
            'UPDRS_3_6a': 'P-S RH',
            'UPDRS_3_7a': 'TT RF',
            'UPDRS_3_8a': 'LA RL',
        },
        'tremor': {
            'UPDRS_3_15a': 'Postural tremor amplitude RA',
            'UPDRS_3_16a': 'Kinetic tremor amplitude RA',
            'UPDRS_3_17a': 'Rest tremor amplitude RA',
            'UPDRS_3_17c': 'Rest tremor amplitude RL',
        }
    },
    'left': {
        'hypokinesia': {
            'UPDRS_3_3c': 'Rigidity LUE',
            'UPDRS_3_3e': 'Rigidity LLE',
            'UPDRS_3_4b': 'FT LH',
            'UPDRS_3_5b': 'Movement LH',
            'UPDRS_3_6b': 'P-S LH',
            'UPDRS_3_7b': 'TT LF',
            'UPDRS_3_8b': 'LA LL',
        },
        'tremor': {
            'UPDRS_3_15b': 'Postural tremor amplitude LA',
            'UPDRS_3_16b': 'Kinetic tremor amplitude LA',
            'UPDRS_3_17b': 'Rest tremor amplitude LA',
            'UPDRS_3_17d': 'Rest tremor amplitude LL',
        }
    }
}

d_updrs_scoring_map = {
    med_stage: {
        'hypokinesia': {
            'right_side': [f'Up3{med_stage}RigRue', f'Up3{med_stage}RigRle'],
            'left_side': [f'Up3{med_stage}RigLle', f'Up3{med_stage}RigLue'],
            'watch_side': [f'Up3{med_stage}LAgiYesDev', f'Up3{med_stage}FiTaYesDev', f'Up3{med_stage}ToTaYesDev', f'Up3{med_stage}ProSYesDev', f'Up3{med_stage}HaMoYesDev'],
            'non_watch_side': [f'Up3{med_stage}HaMoNonDev', f'Up3{med_stage}LAgiNonDev', f'Up3{med_stage}ToTaNonDev', f'Up3{med_stage}FiTaNonDev', f'Up3{med_stage}ProSNonDev']
        },
        'other': [f'Up3{med_stage}Gait', f'Up3{med_stage}Facial', f'Up3{med_stage}RigNec', f'Up3{med_stage}Speech', f'Up3{med_stage}Arise']
    } for med_stage in ['Of', 'On']
}

@dataclass(frozen=True)
class Mappings:
    arm_swing_parameter_map: dict
    metric_map: dict
    activity_map: dict
    segment_map: dict
    segment_rename: dict
    tiers_labels_map: dict
    tiers_rename: dict
    arm_labels_rename: dict
    updrs_3_map: dict
    d_updrs_scoring_map: dict

mappings = Mappings(
    arm_swing_parameter_map = arm_swing_parameter_map,
    metric_map = metric_map,
    activity_map = activity_map,
    segment_map = segment_map,
    segment_rename = segment_rename,
    tiers_labels_map = tiers_labels_map,
    tiers_rename = tiers_rename,
    arm_labels_rename = arm_labels_rename,
    updrs_3_map = updrs_3_map,
    d_updrs_scoring_map = d_updrs_scoring_map
)