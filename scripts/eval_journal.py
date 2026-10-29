"""Append-only, resumable per-item record for long evaluation runs.

`<output>.partial.jsonl` starts with one header line (model, prompt set, generation settings). Each finished item
is appended and flushed immediately, so an interrupted run keeps every completed raw response. Restarting with
the same header skips recorded ids; a different header is refused instead of mixing runs. A torn last line from
a crash is dropped. `finalize` refuses to report unless every expected id is present exactly once, and returns the
rows in prompt order together with failure/truncation counts.
"""
import json
from pathlib import Path


class Journal:
    def __init__(self, output, header):
        self.path = Path(f'{output}.partial.jsonl')
        self.header = header
        self.rows = {}
        self.duplicates = 0
        if self.path.exists():
            lines = self.path.read_text().split('\n')
            if json.loads(lines[0]) != header:
                raise ValueError(f'{self.path} belongs to a different run; use a new output path')
            kept = [lines[0]]
            for line in lines[1:]:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    break  # torn final line from an interrupted write
                if row['id'] in self.rows:
                    self.duplicates += 1
                    continue
                self.rows[row['id']] = row
                kept.append(line)
            self.path.write_text('\n'.join(kept) + '\n')
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(header, ensure_ascii=False) + '\n')

    def done(self, item_id):
        return item_id in self.rows

    def append(self, row):
        if row['id'] in self.rows:
            raise ValueError(f"Duplicate result for {row['id']}")
        with self.path.open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
        self.rows[row['id']] = row

    def finalize(self, ids):
        if len(set(ids)) != len(ids):
            raise ValueError('Prompt ids are not unique')
        missing = [i for i in ids if i not in self.rows]
        extra = sorted(set(self.rows) - set(ids))
        if missing or extra:
            raise ValueError(f'Incomplete run: {len(missing)} missing (first {missing[:3]}), {len(extra)} unexpected')
        rows = [self.rows[i] for i in ids]
        counts = dict(items=len(rows), exceptions=sum(bool(r.get('exception')) for r in rows),
                      skipped_context=sum(bool(r.get('skipped_context')) for r in rows),
                      truncated_output=sum(bool(r.get('truncated_output')) for r in rows),
                      stopped_without_eos=sum(r.get('stopped_on_eos') is False for r in rows),
                      duplicates_dropped_on_resume=self.duplicates)
        return rows, counts
