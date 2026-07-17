#!/bin/sh
set -eu

# Named volumes can retain root ownership from an older deployment. Repair only
# application-owned runtime directories before dropping privileges.
for directory in /app/model_cache /app/session; do
    if [ -d "$directory" ]; then
        chown -R appuser:appuser "$directory"
    fi
done

exec gosu appuser "$@"
