import itertools
import sys
import threading
import time

try:
    # Windows consoles often default to a legacy codepage (e.g. cp1252)
    # that can't encode the emoji used in spinner/agent messages, which
    # crashes this module's background thread with UnicodeEncodeError.
    # Force UTF-8 on stdout; falls back silently if stdout doesn't support
    # reconfigure (e.g. certain redirected/piped contexts).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


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
