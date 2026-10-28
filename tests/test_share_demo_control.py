import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.share_demo_control import tunnel_ready, external_ready


class TunnelHealthTests(unittest.TestCase):
    def test_live_process_without_connections_is_not_ready(self):
        for status, connections, expected in [(503, 0, False), (200, 0, False), (200, 1, True)]:
            with self.subTest(status=status, connections=connections):
                connection = MagicMock()
                connection.getresponse.return_value = SimpleNamespace(
                    status=status, read=lambda: json.dumps({'readyConnections': connections}).encode())
                with patch('scripts.share_demo_control.http.client.HTTPConnection', return_value=connection):
                    self.assertEqual(tunnel_ready(), expected)
                connection.close.assert_called_once()

    def test_external_probe_checks_response_and_keeps_secret_out_of_argv(self):
        for code, body, expected in [(0, '{"ok":true}', True), (0, '{"ok":false}', False), (22, 'error code: 1033', False), (0, 'bad gateway', False)]:
            with self.subTest(code=code, body=body):
                with patch('scripts.share_demo_control.subprocess.run', return_value=SimpleNamespace(returncode=code, stdout=body)) as run:
                    self.assertEqual(external_ready('https://demo.trycloudflare.com', 'sensitive-proxy-key'), expected)
                    self.assertNotIn('sensitive-proxy-key', str(run.call_args.args))
                    self.assertIn('sensitive-proxy-key', run.call_args.kwargs['input'])
                    self.assertIn('--doh-url', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
