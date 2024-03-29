#!/bin/bash

USAGE="$(basename $0) [options|--help]"
PROG=$(basename $0)
ENV_FILE="./bin/env.sh"
if [ ! -f "${ENV_FILE}" ] ; then
    echo "${ENV_FILE} not found; calling from repo root?"
    exit 1
fi
source $ENV_FILE

config_env
${ENV_DIR}/harness.py "$@"
