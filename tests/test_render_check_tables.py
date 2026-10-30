"""Render cross-check of structured 별표 rows (scripts/render_check_tables.py): cell boundaries and row geometry."""
import unittest

from scripts.render_check_tables import cells_in_one_row, numbers_on


class RenderCheckTests(unittest.TestCase):
    def test_numbers_of_adjacent_narrow_cells_do_not_merge(self):
        # a wide 봉급표 draws neighbouring cells closer than one font size apart; the cell change still splits them
        line = [(x, ch, 12, 'c1') for x, ch in zip(range(100, 145, 5), '4,905,100')]
        line += [(x, ch, 12, 'c2') for x, ch in zip(range(146, 191, 5), '4,563,500')]
        self.assertEqual(numbers_on(line), ['4905100', '4563500'])

    def test_cells_must_share_a_row_band_and_keep_column_order(self):
        cells = [(0, 10, 100, 50, 40, 'S등급'), (0, 70, 100, 80, 40, '대상인원의20퍼센트에해당하는인원'),
                 (0, 10, 140, 50, 40, 'A등급'), (0, 70, 140, 80, 40, '대상인원의30퍼센트에해당하는인원')]
        self.assertTrue(cells_in_one_row(['S등급', '대상인원의20퍼센트에해당하는인원'], cells))
        self.assertFalse(cells_in_one_row(['S등급', '대상인원의30퍼센트에해당하는인원'], cells))  # next row's value
        self.assertFalse(cells_in_one_row(['대상인원의20퍼센트에해당하는인원', 'S등급'], cells))  # column order


if __name__ == '__main__':
    unittest.main()
