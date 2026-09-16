from vlnce_baselines.common.opennav_ext.navigation_guidance import NAVIGATION_SYSTEM
from vlnce_baselines.common.opennav_ext.instruction_parsing import ACTION_SYSTEM, LANDMARK_SYSTEM

ACTION_DETECTION = {
    "system": ACTION_SYSTEM,
    "user": "Original navigation instruction:\n{}\nReturn the actions JSON object.",
}

LANDMARK_DETECTION = {
    "system": LANDMARK_SYSTEM,
    "user": "Original navigation instruction:\n{}\nValidated intended actions:\n{}\n"
            "Return the landmarks and terminal_target JSON object.",
}

# Only the selector proposes actions; completion belongs to ACN/terminal checks.
NAVIGATOR = {
    "system": NAVIGATION_SYSTEM,
    "user": "Candidate Viewpoint IDs List: [{}]\nStep: {}\nInstruction: {}\n"
            "Parsed intended actions (not completed actions): {}\nLandmarks: {}\n"
            "Navigation History: {}\nVerified Route Progress: {}\n"
            "Current Environment: {}\nRespond with Thought: ... and Prediction: ...",
}

MOVE_BACK_PROMPT_LINE = (
    "\nMOVE_BACK is available this step: choose it only when returning along the "
    "previous reversible movement to try another branch is more useful than the "
    "listed waypoints. It is not a completion claim. Use exactly MOVE_BACK as the "
    "Prediction value if chosen."
)
