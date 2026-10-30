#!/bin/bash
# SageMaker entrypoint (CPU): independent in-cloud observer of another job's S3 output. Polls WATCH_PREFIX every
# INTERVAL seconds and records every new or changed object (key, size, LastModified, observer time) until DONE_KEY
# exists or TIMEOUT_SECONDS passes; the log is written to the checkpoint path, so it reaches S3 without any local
# process. Read-only on the watched prefix.
#   RUN, WATCH_PREFIX (s3://bucket/prefix/), DONE_KEY (key relative to WATCH_PREFIX), TIMEOUT_SECONDS, INTERVAL
set -euo pipefail
: "${RUN:?}" "${WATCH_PREFIX:?}" "${DONE_KEY:?}"
OUT=/opt/ml/checkpoints/$RUN
mkdir -p "$OUT"
python - "$WATCH_PREFIX" "$DONE_KEY" "${TIMEOUT_SECONDS:-7200}" "${INTERVAL:-30}" "$OUT/observer-log.json" <<'PY'
import json, sys, time, datetime, boto3
prefix, done_key, timeout, interval, log_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
bucket, _, key_prefix = prefix.removeprefix('s3://').partition('/')
s3, seen, events, start = boto3.client('s3'), {}, [], time.time()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
outcome = 'timeout'
while time.time() - start < timeout:
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=key_prefix):
        for o in page.get('Contents', []):
            sig = (o['Size'], o['LastModified'].isoformat())
            if seen.get(o['Key']) != sig:
                events.append({'observed_at': now(), 'key': o['Key'][len(key_prefix):], 'size': o['Size'],
                               'last_modified': o['LastModified'].isoformat()})
                seen[o['Key']] = sig
    json.dump({'watch': prefix, 'started_at': datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
               'events': events, 'outcome': outcome}, open(log_path, 'w'), indent=1)
    if key_prefix + done_key in seen:
        outcome = 'done_marker_seen'
        break
    time.sleep(interval)
done = json.loads(s3.get_object(Bucket=bucket, Key=key_prefix + done_key)['Body'].read()) if outcome == 'done_marker_seen' else None
json.dump({'watch': prefix, 'started_at': datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
           'ended_at': now(), 'outcome': outcome, 'done': done, 'events': events}, open(log_path, 'w'), indent=1)
print(outcome, len(events), 'events')
PY
