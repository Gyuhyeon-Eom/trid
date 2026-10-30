import os
import signal
import subprocess
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import share_demo_control as control  # noqa: E402

# A model server stand-in: ignores SIGTERM (cannot finish its own shutdown) and has a child, like a llama server.
STUCK = ('import signal, subprocess, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
         'subprocess.Popen(["sleep", "60"]); time.sleep(60)')


class ProcessGroupTests(unittest.TestCase):
    def spawn(self):
        p = subprocess.Popen([sys.executable, '-c', STUCK], start_new_session=True)
        self.addCleanup(lambda: control.group_members(p.pid) and os.killpg(p.pid, signal.SIGKILL))
        deadline = time.monotonic() + 5
        while len(control.group_members(p.pid)) < 2 and time.monotonic() < deadline:
            time.sleep(.1)
        return p

    def test_stop_group_ends_a_server_that_ignores_sigterm_and_its_children(self):
        p = self.spawn()
        self.assertEqual(len(control.group_members(p.pid)), 2)
        self.assertEqual(control.stop_group(p.pid, grace=1), [])
        p.wait(timeout=5)
        self.assertEqual(control.group_members(p.pid), [])  # the child (llama server stand-in) is gone too

    def test_recorded_group_counts_only_with_the_recorded_leader_start_time(self):
        p = self.spawn()
        leader = next(m for m in control.group_members(p.pid) if m[0] == p.pid)
        self.assertEqual(len(control.owned_group({'backend_pgid': p.pid, 'backend_started': leader[1]})), 2)
        self.assertEqual(control.owned_group({'backend_pgid': p.pid, 'backend_started': 'Thu Jan  1 00:00:00 1970'}), [])
        self.assertEqual(control.owned_group({}), [])


if __name__ == '__main__':
    unittest.main()
