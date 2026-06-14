from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "models" / "cheat_detector.pkl"
REPORTS_DIR  = PROJECT_ROOT / "reports"


N_TRAIN      = 5000
N_TEST       = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42


RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 10


PASS_THRESHOLD       = 0.3
TRIGGERBOT_THRESHOLD = 0.7


VERDICT_COLORS = {
    "PASS":       "#2ecc71",
    "REVIEW":     "#f39c12",
    "TRIGGERBOT": "#e74c3c",
}
