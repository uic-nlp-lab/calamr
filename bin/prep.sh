#!/bin/bash

USAGE="$(basename $0) <mismatchcorp|mergeanons|downmodel|parsecorp|clean[all,world]>"
PROG=$(basename $0)
ENV_FILE="./bin/env.sh"
if [ ! -f "${ENV_FILE}" ] ; then
    echo "${ENV_FILE} not found; calling from repo root?"
    exit 1
fi
source $ENV_FILE

case "$1" in
    mergeanons)
	config_env
	exec ${PY_INST_BIN} ${ENV_ROOT_DIR}/src/bin/merge-proxy-anons.py
	;;

    mismatchcorp)
	config_env
	exec ${PY_INST_BIN} ${ENV_ROOT_DIR}/src/bin/mismatch-corp.py
	;;

    parsecorp)
	config_env
	exec ${PY_INST_BIN} ${ENV_ROOT_DIR}/src/bin/parse-corp.py
	parsed_dir=${ENV_ROOT_DIR}/corpus/parsed
	cp -r ${ENV_ROOT_DIR}/corpus/amr-rel/amr_annotation_3.0/data/merge/parsed/proxy-report $parsed_dir
	;;

    score)
	config_env
	exec ${PY_INST_BIN} ${ENV_ROOT_DIR}/src/bin/score.py
	;;

    align)
	config_env
	for corpus in proxy-report proxy-report-1_0 ; do
	    for reentry_type in fix only-report; do \
		for split in training test dev ; do \
		    echo "aligning split: $split, reentry: $reentry_type"
		    exec ${PY_INST_BIN} \
			 ${ENV_ROOT_DIR}/src/bin/align.py \
			 $corpus $split $reentry_type
		done
	    done
	done
	;;

    alignstats)
	config_env
	exec ${PY_INST_BIN} ${ENV_ROOT_DIR}/src/bin/align-report.py \
	     -results results/aligns
	;;

    clean)
	clean
	;;

    cleanall)
	clean
	clean_corpus
	clean_results
	;;

    cleanworld)
	clean
	clean_corpus
	clean_env
	;;

    *)
	bail "no action given"
	;;
esac

