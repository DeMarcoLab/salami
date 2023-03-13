# import time module, Observer, FileSystemEventHandler
import os
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from salami import config as cfg
from salami.segmentation import segmentation as sseg
from fibsem import utils

class SalamiWatcher:

    def __init__(self, path_to_watch: Path, protocol_path: Path):
        self.observer = Observer()
        self.path_to_watch = path_to_watch
        self.protocol = utils.load_yaml(protocol_path)
    def run(self):
        event_handler = Handler(path=self.path_to_watch, protocol=self.protocol)
        self.observer.schedule(event_handler, self.path_to_watch, recursive = True)
        self.observer.start()
        print(f"Observer watching directory: {self.path_to_watch}")
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()


from salami.denoise import inference as denoise_inference

class Handler(FileSystemEventHandler):
	
    def __init__(self, path, protocol) -> None:
        super().__init__()

        self.protocol = protocol
        self.model = sseg.load_model(None)
        self.dmodel = denoise_inference.setup_denoise_inference(self.protocol["denoise"])

        self.RAW_PATH = path
        path = os.path.dirname(path) 
        self.DENOISE_PATH = os.path.join(path, cfg.DENOISE_DIR)
        self.SEG_PATH = os.path.join(path, cfg.SEG_DIR)

        os.makedirs(self.DENOISE_PATH, exist_ok=True)
        os.makedirs(self.SEG_PATH, exist_ok=True)
    
    def on_created(self, event):
        
        # time how long each step takes
        t0 = time.time()

        # Event is created, you can process it now
        print("Watchdog received created event - % s." % event.src_path)

        if not event.src_path.endswith(".tif"):
            return

        t1 = time.time()
        # run denoising
        denoise_inference.run_denoise_step(
            trainer=self.dmodel,
            fname=event.src_path,
            input_path=self.RAW_PATH,
            output_path=self.DENOISE_PATH,
        )

        t2 = time.time()
        # run segmentation
        # TODO: this should only go off of the denoised image

        if os.path.exists(os.path.join(self.DENOISE_PATH, os.path.basename(event.src_path))):
            sseg.run_segmentation_step(model = self.model, 
                fname=event.src_path, 
                output_path=self.SEG_PATH)


        t3 = time.time()

        # print time for each step
        basename = os.path.basename(event.src_path).split(".")[0]
        print(f"PIPELINE {basename} | DENOISE {t2-t1:.3f}s | SEGMENT {t3-t2:.3f}s | TOTAL {t3-t0:.3f}s")

def main():
    protocol_path = "/home/patrick/github/salami/salami/protocol/protocol.yaml"
    watch_dir = "/home/patrick/github/salami/salami/output2/raw"
    os.makedirs(watch_dir, exist_ok=True)
    watch = SalamiWatcher(watch_dir, protocol_path=protocol_path)
    watch.run()


if __name__ == '__main__':
    main()

#https://www.geeksforgeeks.org/create-a-watchdog-in-python-to-look-for-filesystem-changes/