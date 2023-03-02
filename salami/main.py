
import os
from fibsem import utils
from salami import structures as st

def run_salami_analysis(path):

    microscope, settings = utils.setup_session()
    
    path = os.path.join(settings.image.save_path, "data")

    df = st.create_sweep_parameters(settings, None)

    st.run_sweep_collection(microscope, settings, break_idx=10)
    df = st.run_sweep_analysis(path)
    df = st.join_df(path)
    st.plot_metrics(path)

def full_pipeline():

    import random
    import time
    print("Hello Denoising Pipeline")

    time.sleep(random.randint(1, 5))
    print("Loading data...")
    time.sleep(random.randint(1, 5))
    print("Preprocessing data...")
    time.sleep(random.randint(1, 5))
    print("Running denoising model...")
    time.sleep(random.randint(1, 5))
    print("Saving results...")
    time.sleep(random.randint(1, 5))
    print("Restacking and aligning arrays...")
    time.sleep(random.randint(1, 5))
    print("Running segmentation model...")
    time.sleep(random.randint(1, 5))
    print("Saving results...")


def main():
    # run_salami_analysis("data")

    full_pipeline()


if __name__ == "__main__":
    main()