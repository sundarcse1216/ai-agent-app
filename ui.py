import itertools
import sys
import threading
import time


class Spinner:

    def __init__(self, message="Thinking"):
        self.message = message
        self.running = False

    def start(self):
        self.running = True

        def animate():
            for c in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
                if not self.running:
                    break
                sys.stdout.write(f"\r{self.message} {c}")
                sys.stdout.flush()
                time.sleep(0.1)

            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

        self.thread = threading.Thread(target=animate)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
