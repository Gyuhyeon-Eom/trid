"""Throwaway B-009, real S3: (1) conditional put on a scratch key refuses to replace bytes; (2) fetch under a changed
clean version against the real 22nd-term index makes 0 HTTP calls and leaves every stored raw object unchanged."""
import json
import sys
from unittest import mock

sys.path.insert(0, '/Users/eomgyuhyeon/사업/trid/scripts')
sys.path.insert(0, '/Users/eomgyuhyeon/사업/trid')
import collect_assembly_minutes as am  # noqa: E402

s3 = am.s3_client()
out = {}

# (1) scratch key in B's state prefix, removed afterwards
key = 'corpus/state/assembly-minutes/probe-b009/put-new.bin'
first = am.put_new(s3, key, b'first')
etag = s3.head_object(Bucket=am.BUCKET, Key=key)['ETag']
try:
    am.put_new(s3, key, b'second')
    refused = False
except am.RawExists:
    refused = True
body = s3.get_object(Bucket=am.BUCKET, Key=key)['Body'].read()
same_again = am.put_new(s3, key, b'first') == first
out['conditional_put'] = {'different_bytes_refused': refused, 'bytes_after': body.decode(),
                          'etag_unchanged': s3.head_object(Bucket=am.BUCKET, Key=key)['ETag'] == etag,
                          'same_bytes_accepted_without_write': same_again}
s3.delete_object(Bucket=am.BUCKET, Key=key)


# (2) version change, real index, HTTP forbidden
def snapshot():
    snap = {}
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=am.BUCKET, Prefix='corpus/raw/assembly-minutes/'):
        for o in page.get('Contents', []):
            snap[o['Key']] = (o['ETag'], o['LastModified'].isoformat(), o['Size'])
    return snap


before = snapshot()
log = '/Users/eomgyuhyeon/사업/trid/logs/requests-record.assembly.go.kr.jsonl'
log_lines = open(log).read().count('\n')
calls = []
client = am.Client(am.HostLimiter(1.0, host='record.assembly.go.kr'), log)
client.session.request = lambda *a, **kw: calls.append(a) or (_ for _ in ()).throw(RuntimeError('HTTP forbidden in probe'))
with mock.patch.object(am, 'VERSION', 'clean-v99-probe'):
    am.fetch(client, s3, 'pilot22-index.json', 100, 1, None, am.Metrics(), '20261030', {'1', '2', '5'})
after = snapshot()
out['version_change_fetch'] = {'version': 'clean-v99-probe', 'http_calls': len(calls),
                               'raw_objects_before': len(before), 'raw_objects_after': len(after),
                               'raw_objects_changed': sum(before[k] != after.get(k) for k in before),
                               'request_log_lines_added': open(log).read().count('\n') - log_lines,
                               'probe_version_objects': len(am.list_keys(s3, 'corpus/state/assembly-minutes/clean-v99-probe/'))
                                                        + len(am.list_keys(s3, 'corpus/clean/assembly-minutes/clean-v99-probe/'))}
print(json.dumps(out, indent=1))
json.dump(out, open('/tmp/trid-corpus/qa/proof-raw-protect.json', 'w'), indent=1)
