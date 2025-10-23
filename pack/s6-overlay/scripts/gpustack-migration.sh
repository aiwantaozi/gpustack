#!/command/with-contenv /bin/bash
# shellcheck disable=SC1008
# ================================
# GPUStack migration oneshot service
# ================================

if [ -n "${GPUSTACK_MIGRATION_DATA_DIR}" ]; then
    echo "[INFO] Using GPUSTACK_MIGRATION_DATA_DIR: ${GPUSTACK_MIGRATION_DATA_DIR}."

    DB_PORT=5432
    if [ -n "$POSTGRES_PORT" ]; then
        DB_PORT="$POSTGRES_PORT"
    fi
    exec gpustack migrate --migration-data-dir "${GPUSTACK_MIGRATION_DATA_DIR}" --database-url "postgresql://root@localhost:${DB_PORT}/gpustack"
else
    echo "[INFO] No migration data dir specified, skipping."
fi
