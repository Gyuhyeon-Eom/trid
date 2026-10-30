"""The HWPX image-strip fallback must not loosen the zip limits it works around (B-026)."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import scripts.collect_admrul_attachments as m


def build(path, entries, images=0, encrypted=()):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, body in entries:
            z.writestr(name, body)
        for i in range(images):
            z.writestr(f'BinData/image{i}.png', b'\x89PNG' + bytes([i % 251]) * 64)
    raw = bytearray(path.read_bytes())
    for name in encrypted:  # zipfile cannot write the bit; set it in the local and central headers
        enc = name.encode()
        for sig, flag_at, name_at in ((b'PK\x03\x04', 6, 30), (b'PK\x01\x02', 8, 46)):
            pos = raw.find(sig)
            while pos >= 0 and raw[pos + name_at:pos + name_at + len(enc)] != enc:
                pos = raw.find(sig, pos + 4)
            raw[pos + flag_at] |= 0x1
    path.write_bytes(bytes(raw))


BODY = [('mimetype', b'application/hwp+zip'), ('Contents/section0.xml', b'<p>' + '본문 '.encode() * 2000 + b'</p>')]


class StripHwpxImagesTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.src, self.dst = self.d / 'in.hwpx', self.d / 'out.hwpx'

    def refused(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            m.strip_hwpx_images(self.src, self.dst)

    def test_image_heavy_file_keeps_every_non_image_entry_unchanged(self):
        build(self.src, BODY, images=1200)
        info = m.strip_hwpx_images(self.src, self.dst)
        self.assertEqual((info['images_removed'], info['entries_kept']), (1200, 2))
        with zipfile.ZipFile(self.dst) as z:
            self.assertEqual(z.namelist(), ['mimetype', 'Contents/section0.xml'])
            self.assertEqual(z.read('Contents/section0.xml'), BODY[1][1])

    def test_highly_compressed_entry_is_refused(self):
        build(self.src, BODY + [('Contents/section1.xml', b'\0' * (8 * 1024 * 1024))], images=600)
        self.refused('compression ratio')

    def test_unsafe_paths_are_refused(self):
        for name in ('../evil.xml', '/etc/evil.xml', 'Contents\\evil.xml', 'C:/evil.xml', 'Contents/../../evil.xml'):
            with self.subTest(name=name):
                build(self.src, BODY + [(name, b'x')], images=600)
                self.refused('unsafe zip entry path')

    def test_encrypted_entry_is_refused(self):
        build(self.src, BODY, images=600, encrypted=('Contents/section0.xml',))
        self.refused('encrypted')

    def test_duplicate_names_are_refused(self):
        build(self.src, BODY + [('Contents/section0.xml', b'<p>other</p>')], images=600)
        self.refused('duplicate')

    def test_kordoc_limits_still_apply_to_the_original(self):
        build(self.src, BODY + [(f'Contents/s{i}.xml', b'x') for i in range(m.HWPX_MAX_KEPT_ENTRIES)], images=10)
        self.refused('non-image zip entries')
        build(self.src, BODY, images=600)
        with mock.patch.object(m, 'HWPX_MAX_ENTRIES', 500):
            self.refused('zip entries 602')
        with mock.patch.object(m, 'HWPX_MAX_UNCOMPRESSED', 40_000):  # image bytes count towards the total
            self.refused('declared uncompressed total')


if __name__ == '__main__':
    unittest.main()
