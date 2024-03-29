#!/bin/bash

USAGE="$(basename $0) <createcorp>"
PROG=$(basename $0)
ENV_FILE="./bin/env.sh"
if [ ! -f "${ENV_FILE}" ] ; then
    echo "${ENV_FILE} not found; calling from repo root?"
    exit 1
fi
source $ENV_FILE

case "$1" in
    # create the micro corpus from corpus/amr-micro-summary.json
    createcorp)
	config_env
	exec ${PY_INST_BIN} ${ENV_DIR}/harness.py -c etc/micro.conf micro
	;;

    align)
	config_env
	exec ${PY_INST_BIN} ${ENV_DIR}/harness.py \
	     -c etc/micro.conf aligncorp ALL -f txt -r 5 -d align-micro
	;;

    *)
	bail "no action given"
	;;
esac

