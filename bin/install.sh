#!/bin/bash

USAGE="$(basename $0) <python home directory>"
ENV_FILE="./bin/env.sh"
if [ ! -f "${ENV_FILE}" ] ; then
    echo "${ENV_FILE} not found; calling from repo root?"
    exit 1
fi
PY_HOME=$1 ; shift
source $ENV_FILE
install_env
