import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from plutocontrol import pluto

# Initialize drone connection
drone = pluto()
drone.connect()

# Initialize HandDetector from cvzone with higher confidence detection
detector = HandDetector(detectionCon=0.9, maxHands=2)  # Detect up to 2 hands

# Function to determine if the palm or back side of the hand is facing the camera
def get_hand_orientation(lmList, handType):
    if len(lmList) == 21:
        wrist = np.array(lmList[0])
        thumb_tip = np.array(lmList[4])
        index_tip = np.array(lmList[8])
        middle_tip = np.array(lmList[12])
        
        vector_thumb = thumb_tip - wrist
        vector_index = index_tip - wrist
        vector_middle = middle_tip - wrist
        
        palm_normal = np.cross(vector_thumb, vector_index)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        camera_axis = np.array([0, 0, 1])
        dot_product = np.dot(palm_normal, camera_axis)
        
        if handType == "Right":
            dot_product = -dot_product
        
        if dot_product > 0.7:
            return "Palm Facing Camera"
        else:
            return "Back Facing Camera"
    return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

# Variables to store the last printed command
last_command = None

# Variables to keep track of command states
command_states = {"arm": False, "disarm": False, "take off": False, "land": False}

# Flags to ensure only one command per frame
command_executed = {"arm": False, "disarm": False, "take off": False, "land": False}

while True:
    
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    num_hands = len(hands)

    # Reset flags for the new frame
    for key in command_executed:
        command_executed[key] = False

    if num_hands > 0:
        orientations = []
        for hand in hands:
            lmList = hand['lmList']
            handType = hand['type']
            orientation = get_hand_orientation(lmList, handType)
            orientations.append(orientation)
        
        current_command = None

        if num_hands == 2:
            if all(o == "Palm Facing Camera" for o in orientations):
                if not command_states["arm"] and not command_executed["arm"]:
                    current_command = "arm"
                    drone.arm()
                    command_states["arm"] = True
                    command_states["disarm"] = False
                    command_states["take off"] = False
                    command_states["land"] = False
                    command_executed["arm"] = True
            elif all(o == "Back Facing Camera" for o in orientations):
                if not command_states["disarm"] and not command_executed["disarm"]:
                    current_command = "disarm"
                    drone.disarm()
                    command_states["disarm"] = True
                    command_states["arm"] = False
                    command_states["take off"] = False
                    command_states["land"] = False
                    command_executed["disarm"] = True
        elif num_hands == 1:
            right_hand = next((hand for hand in hands if hand['type'] == 'Right'), None)
            left_hand = next((hand for hand in hands if hand['type'] == 'Left'), None)
            
            if right_hand:
                orientation = get_hand_orientation(right_hand['lmList'], 'Right')
                if orientation == "Back Facing Camera":
                    if command_states["arm"] and not command_states["take off"] and not command_executed["take off"]:
                        current_command = "take off"
                        drone.take_off()
                        command_states["take off"] = True
                        command_states["arm"] = False
                        command_states["disarm"] = False
                        command_states["land"] = False
                        command_executed["take off"] = True
                    elif not command_states["arm"]:
                        current_command = None
            elif left_hand:
                orientation = get_hand_orientation(left_hand['lmList'], 'Left')
                if orientation == "Back Facing Camera":
                    if command_states["take off"] and not command_states["land"] and not command_executed["land"]:
                        current_command = "land"
                        drone.land()
                        command_states["land"] = True
                        command_states["arm"] = False
                        command_states["disarm"] = False
                        command_states["take off"] = False
                        command_executed["land"] = True
                    elif not command_states["take off"]:
                        current_command = None
                else:
                    command_states["land"] = False

        # Print the command only if it's different from the last printed command
        if current_command and current_command != last_command:
            print(f"Executing command: {current_command}")
            last_command = current_command

        # Display messages on the frame
        for i, hand in enumerate(hands):
            message = f"{hand['type'].capitalize()} Hand: {orientations[i]}"
            cv2.putText(img, message, (50, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "Please show at least two hands", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
