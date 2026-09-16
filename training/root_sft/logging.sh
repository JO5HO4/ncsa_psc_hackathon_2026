# Sourced after RUN_DIR has been created. Captures all subsequent run output.
# Does not log environment variables or expand commands with shell tracing.
export PYTHONUNBUFFERED=1
ROOT_RUN_STAGE=initialization
ROOT_RUN_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ROOT_RUN_SECONDS=$SECONDS
exec 3>&1 4>&2
exec > >(tee "$RUN_DIR/run.log") 2>&1
ROOT_LOG_PID=$!

root_stage() {
  ROOT_RUN_STAGE="$1"
  printf '\n[%s] STAGE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ROOT_RUN_STAGE"
}

root_log_finish() {
  local code=$?
  trap - EXIT ERR
  local ended elapsed state
  ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed=$((SECONDS - ROOT_RUN_SECONDS))
  state=failed
  [[ "$code" == 0 ]] && state=success
  printf '\n[%s] RUN %s stage=%s exit_code=%s elapsed_seconds=%s\n' \
    "$ended" "$state" "$ROOT_RUN_STAGE" "$code" "$elapsed"
  # Close the pipe and wait for tee so the last message is flushed before exit.
  exec 1>&3 2>&4 3>&- 4>&-
  if ! wait "$ROOT_LOG_PID"; then
    printf 'Log writer failed; run.log may be incomplete.\n' >&2
    [[ "$code" != 0 ]] || code=1
    state=failed
  fi
  printf '{"status":"%s","stage":"%s","exit_code":%s,"started_utc":"%s","ended_utc":"%s","elapsed_seconds":%s}\n' \
    "$state" "$ROOT_RUN_STAGE" "$code" "$ROOT_RUN_STARTED" "$ended" "$elapsed" > "$RUN_DIR/status.json"
  exit "$code"
}
trap root_log_finish EXIT
trap 'printf "ERROR stage=%s line=%s exit_code=%s\n" "$ROOT_RUN_STAGE" "$LINENO" "$?" >&2' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
printf '[%s] RUN started\n' "$ROOT_RUN_STARTED"
