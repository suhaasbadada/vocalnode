#!/bin/bash
# Read the tool input payload provided by Claude Code
PAYLOAD=$(cat)

if echo "$PAYLOAD" | grep -qE "apt-get|brew install|alsa|pulseaudio"; then
    echo "BLOCKED: System-level package installation or audio driver modification is forbidden. Write dependencies to system_reqs.txt instead."
    exit 2
fi

exit 0