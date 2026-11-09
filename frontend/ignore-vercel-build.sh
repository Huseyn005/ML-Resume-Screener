#!/bin/sh

previous_sha=${VERCEL_GIT_PREVIOUS_SHA:-}
commit_sha=${VERCEL_GIT_COMMIT_SHA:-}

if [ -z "$previous_sha" ] || [ -z "$commit_sha" ] || [ "$previous_sha" = "0000000000000000000000000000000000000000" ]; then
  exit 1
fi

case "$1" in
  root)
    if git diff --quiet "$previous_sha" "$commit_sha" -- frontend/ vercel.json; then
      exit 0
    fi
    ;;
  frontend)
    if git diff --quiet "$previous_sha" "$commit_sha" -- .; then
      exit 0
    fi
    ;;
  *)
    exit 1
    ;;
esac

exit 1
