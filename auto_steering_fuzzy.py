"""
Auto Steering with SVM + Fuzzy Logic + Keyboard Control
-------------------------------------------------------
# 7b. auto_steering_fuzzy.py

Pipeline:
    Frame -> ROI -> SVM (label, confidence)
         -> Fuzzy Controller (steering angle)
         -> Angle -> Action
         -> Keyboard LEFT/RIGHT -> Game
"""
#%pip install keyboard
#%pip install scikit-fuzzy


import time
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import keyboard  # pip install keyboard

# =======================================================
# 1. FUZZY VARIABLES AND MEMBERSHIP FUNCTIONS
# =======================================================

# Universes
direction_universe = np.arange(-1.0, 1.01, 0.01)   # -1 to 1
confidence_universe = np.arange(0.0, 1.01, 0.01)   # 0 to 1
steering_universe = np.arange(-35.0, 35.1, 0.1)    # -35° to +35°

# Fuzzy variables
direction = ctrl.Antecedent(direction_universe, 'direction')
confidence = ctrl.Antecedent(confidence_universe, 'confidence')
steering = ctrl.Consequent(steering_universe, 'steering')

# --- Direction membership functions ---
direction['strong_left']  = fuzz.trimf(direction_universe, [-1.0, -1.0, -0.7])
direction['left']         = fuzz.trimf(direction_universe, [-0.9, -0.6, -0.3])
direction['slight_left']  = fuzz.trimf(direction_universe, [-0.4, -0.2,  0.0])
direction['center']       = fuzz.trimf(direction_universe, [-0.1,  0.0,  0.1])
direction['slight_right'] = fuzz.trimf(direction_universe, [ 0.0,  0.2,  0.4])
direction['right']        = fuzz.trimf(direction_universe, [ 0.3,  0.6,  0.9])
direction['strong_right'] = fuzz.trimf(direction_universe, [ 0.7,  1.0,  1.0])

# --- Confidence membership functions ---
confidence['low']    = fuzz.trimf(confidence_universe, [0.0, 0.0, 0.4])
confidence['medium'] = fuzz.trimf(confidence_universe, [0.2, 0.5, 0.8])
confidence['high']   = fuzz.trimf(confidence_universe, [0.6, 1.0, 1.0])

# --- Steering output membership functions ---
steering['strong_left']  = fuzz.trimf(steering_universe, [-35, -35, -30])
steering['left']         = fuzz.trimf(steering_universe, [-32, -25, -15])
steering['slight_left']  = fuzz.trimf(steering_universe, [-18, -10,  -5])
steering['center']       = fuzz.trimf(steering_universe, [ -5,   0,   5])
steering['slight_right'] = fuzz.trimf(steering_universe, [  5,  10,  18])
steering['right']        = fuzz.trimf(steering_universe, [ 15,  25,  32])
steering['strong_right'] = fuzz.trimf(steering_universe, [ 30,  35,  35])

# =======================================================
# 2. RULE BASE
# =======================================================

# Basic direction -> steering rules
rule1 = ctrl.Rule(direction['strong_left'],  steering['strong_left'])
rule2 = ctrl.Rule(direction['left'],         steering['left'])
rule3 = ctrl.Rule(direction['slight_left'],  steering['slight_left'])
rule4 = ctrl.Rule(direction['center'],       steering['center'])
rule5 = ctrl.Rule(direction['slight_right'], steering['slight_right'])
rule6 = ctrl.Rule(direction['right'],        steering['right'])
rule7 = ctrl.Rule(direction['strong_right'], steering['strong_right'])

# Confidence refinement rules
rule8  = ctrl.Rule(direction['left']  & confidence['low'],
                   steering['slight_left'])
rule9  = ctrl.Rule(direction['left']  & confidence['high'],
                   steering['left'])

rule10 = ctrl.Rule(direction['right'] & confidence['low'],
                   steering['slight_right'])
rule11 = ctrl.Rule(direction['right'] & confidence['high'],
                   steering['right'])

# Build control system
steering_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6, rule7,
    rule8, rule9, rule10, rule11
])

# =======================================================
# 3. HELPER FUNCTIONS (FUZZY + ACTIONS)
# =======================================================

def encode_direction(label: str) -> float:
    """
    Map string label from SVM to numeric direction in [-1, 1].
    """
    label = label.lower()
    if label == 'left':
        return -1.0
    elif label == 'right':
        return 1.0
    else:  # straight or unknown
        return 0.0


def fuzzy_steering_from_svm(label: str, conf: float) -> float:
    """
    Compute steering angle from SVM label and confidence using fuzzy logic.

    Parameters
    ----------
    label : str
        'Left', 'Right', or 'Straight' (case-insensitive).
    conf : float
        Confidence in [0, 1]. If not available, you can pass 0.7.

    Returns
    -------
    float
        Steering angle in degrees (negative = left, positive = right).
    """
    d_val = encode_direction(label)
    c_val = float(np.clip(conf, 0.0, 1.0))

    # Fresh simulation each call (simple and safe)
    sim = ctrl.ControlSystemSimulation(steering_ctrl)
    sim.input['direction'] = d_val
    sim.input['confidence'] = c_val
    sim.compute()

    angle = sim.output['steering']
    return float(angle)


def steering_action_from_angle(angle: float) -> str:
    """
    Map steering angle (in degrees) to a discrete steering action.

    Returns:
        'hard_left', 'soft_left', 'straight', 'soft_right', 'hard_right'
    """
    if angle < -20:
        return 'hard_left'
    elif -20 <= angle < -5:
        return 'soft_left'
    elif -5 <= angle <= 5:
        return 'straight'
    elif 5 < angle <= 20:
        return 'soft_right'
    else:  # angle > 20
        return 'hard_right'


def apply_steering_action(action: str,
                          duration_soft: float = 0.05,
                          duration_hard: float = 0.15):
    """
    Simulate key presses for steering based on the action string.

    duration_soft : how long to press for soft turns (seconds)
    duration_hard : how long to press for hard turns (seconds)
    """
    if action == 'hard_left':
        keyboard.press('left')
        time.sleep(duration_hard)
        keyboard.release('left')

    elif action == 'soft_left':
        keyboard.press('left')
        time.sleep(duration_soft)
        keyboard.release('left')

    elif action == 'straight':
        # release both to avoid being stuck
        keyboard.release('left')
        keyboard.release('right')

    elif action == 'soft_right':
        keyboard.press('right')
        time.sleep(duration_soft)
        keyboard.release('right')

    elif action == 'hard_right':
        keyboard.press('right')
        time.sleep(duration_hard)
        keyboard.release('right')


# =======================================================
# 4. PLACEHOLDER: SVM PREDICTION (YOU PLUG YOUR MODEL HERE)
# =======================================================

def svm_predict_direction_and_confidence(roi):
    """
    TODO: Replace this with your real SVM model prediction.

    roi : preprocessed Region of Interest (image features)

    Returns:
        label : 'Left', 'Straight', or 'Right'
        conf  : float in [0, 1] (confidence)

    For now, this is just a dummy example.
    """
    # Example dummy behavior (for testing the fuzzy + keyboard pipeline):
    # You should replace this with something like:
    # label = svm_model.predict(roi)[0]
    # prob  = svm_model.predict_proba(roi)[0]
    # conf  = float(np.max(prob))
    #
    # return label, conf
    return 'Left', 0.8  # dummy fixed output


# =======================================================
# 5. DEMO LOOP (NO CAMERA, JUST PRINT + OPTIONAL KEYBOARD)
# =======================================================

def demo_without_camera():
    """
    Simple demo that pretends we got some SVM outputs and shows how
    fuzzy + keyboard would work.
    """
    test_data = [
        ('Left', 0.9),
        ('Left', 0.3),
        ('Straight', 0.6),
        ('Right', 0.2),
        ('Right', 0.95),
    ]

    for label, conf in test_data:
        angle = fuzzy_steering_from_svm(label, conf)
        action = steering_action_from_angle(angle)

        print(f"SVM: {label:8s}, conf={conf:.2f}  ->  "
              f"angle={angle:6.2f}°  ->  action={action}")

        # Uncomment below to actually press keys:
        # apply_steering_action(action)

        time.sleep(0.3)


# =======================================================
# 6. OUTLINE FOR REAL-TIME LOOP (YOU ADD FRAME CAPTURE)
# =======================================================

def realtime_loop():
    """
    Outline for the real-time loop.
    You must:
      - Grab frame from screen or webcam
      - Crop ROI, resize, and preprocess for SVM
      - Call svm_predict_direction_and_confidence(roi)
    """
    while True:
        # TODO: Replace this with your real frame/ROI pipeline.
        # Example:
        # frame = grab_frame()
        # roi = preprocess(frame)

        roi = None  # placeholder
        label, conf = svm_predict_direction_and_confidence(roi)

        angle = fuzzy_steering_from_svm(label, conf)
        action = steering_action_from_angle(angle)

        print(f"SVM={label}, conf={conf:.2f}, angle={angle:.2f}, action={action}")

        # Uncomment to actually control the game:
        # apply_steering_action(action)

        time.sleep(0.05)  # control frequency


# =======================================================
# 7. MAIN
# =======================================================

if __name__ == "__main__":
    # First, test fuzzy + mapping without keyboard:
    demo_without_camera()

    # Later, when ready, comment demo and run realtime_loop():
    # realtime_loop()
