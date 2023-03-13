import os


SALAMI_PATH = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(SALAMI_PATH, "protocol", "protocol.yaml")
SWEEP_PATH = os.path.join(SALAMI_PATH, "protocol", "sweep.yaml")


DATA_DIR = "data"
RAW_DIR = "raw"
SEG_DIR = "seg"
DENOISE_DIR = "denoise"


LABELS =[
            "Background",
            "Nucleus",
            "Nucleolus",
            "Cytoplasm",
            "Mitochondria",
            "ER",
            "Golgi",
        ]