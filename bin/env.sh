#!/bin/bash

# environment
PROG=$(basename $0)
ENV_ROOT_DIR=$(pwd)
ENV_DIR=${ENV_ROOT_DIR}/bin
PY_LIB_DIR=${ENV_ROOT_DIR}/lib
AMR_COREF_DIR=${PY_LIB_DIR}/amr_coref
PY_INST_DIR=pyenv
PY_INST_BIN=${PY_INST_DIR}/bin/python3


if [ ! -d "${PY_LIB_DIR}/trans" ] ; then
    echo "$PROG: library not found, calling $PROG from repo root?"
    exit 1
fi


## Utility functions
#
function prhead() {
    echo "--------------------${1}:"
}

function bail() {
    msg=$1 ; shift
    echo "usage: $PROG $USAGE"
    echo "$PROG: error: $msg"
    exit 1
}

function parse_args() {
    if [ -z "$PY_HOME" ] ; then
	bail "python installation directory not set"
    fi
    if [ ! -d $PY_HOME ] ; then
	bail "python home should be a directory: $PY_HOME"
    fi
    PY_BIN=${PY_HOME}/bin/python3
    if [ ! -x $PY_BIN ] ; then
	bail "No python executable: ${PY_BIN}"
    fi
    PY_VER=$($PY_BIN --version)
    if [ -z "$PY_VER" ] ; then
	bail "Python is not executable: ${PY_BIN}"
    fi
    export PATH=${PY_HOME}/bin:${PATH}
}


## Install functions
#
function install_amr_coref() {
    if [ -d "${AMR_COREF_DIR}" ] ; then
	echo "amr_coref already installed"
    else
	echo "installing amr_coref..."
	( mkdir -p $PY_LIB_DIR &&
	      cd $PY_LIB_DIR &&
	      git clone https://github.com/bjascob/amr_coref &&
	      cd $(basename $AMR_COREF_DIR) &&
	      git co 5c799171fd8866417798559851a06cea2fadb510 &&
	      git apply ../amr-coref.patch &&
	      git status
	)
    fi
}

function install_python() {
    if [ -d "$PY_INST_DIR" ] ; then
	echo "python virtual environment already installed"
    else
	prhead "python"
	echo "Python version should be version 3.10.x, but not checking"
	echo "installing python version ${PY_VER} from ${PY_HOME}..."
	${PY_BIN} -m venv --copies ${PY_INST_DIR}
    fi
}

function install_deps() {
    pkg_count=$(${PY_INST_BIN} -m pip freeze | wc -l)
    echo "found $pkg_count packages"
    if [ "$pkg_count" -ne 0 ]; then
	echo "dependencies seem to already be installed"
    else
	prhead "python dependencies"
	${PY_INST_BIN} -m pip install -r src/python/requirements.txt
    fi
}

function install_env() {
    parse_args
    prhead "installing calamr"
    install_amr_coref
    install_python
    install_deps
}

function clean() {
    prhead "cleaning"
    for dir in ${ENV_ROOT_DIR}/data ${ENV_ROOT_DIR}/target ; do
	if [ -d $dir ] ; then
	    echo "removing $dir..."
	    rm -rf $dir
	fi
    done
    rm -f ${ENV_ROOT_DIR}/*.log scores.csv
}

function clean_corpus() {
    rm -f ${ENV_ROOT_DIR}/corpus/amr-rel/amr-mismatch-proxy.txt
    if [ -d ${ENV_ROOT_DIR}/corpus ] ; then
	rm -r ${ENV_ROOT_DIR}/corpus
	git co ${ENV_ROOT_DIR}/corpus
    fi
}

function clean_results() {
    if [ -d ${ENV_ROOT_DIR}/results ] ; then
	rm -r ${ENV_ROOT_DIR}/results
	git co ${ENV_ROOT_DIR}/results
    fi
}

function clean_env() {
    prhead "remove Python environment"
    for dir in ${PY_INST_DIR} ${AMR_COREF_DIR} ; do
	if [ -d $dir ] ; then
	    echo "removing $dir..."
	    rm -rf $dir
	fi
    done
}


## Entry point config functions
#
function config_env() {
    CALAMR_PY_HOME=$(pwd)/pyenv
    CALAMR_PY_BIN=${CALAMR_PY_HOME}/bin/python3
    if [ ! -x "${CALAMR_PY_BIN}" ] ; then
	echo "$PROG: wrong working directory or need to run ./bin/install.sh"
	exit 1
    fi
    export PATH=${CALAMR_PY_HOME}/bin:${PATH}
    export PYTHONPATH="\
${ENV_ROOT_DIR}/src/python:\
${PY_LIB_DIR}/trans:\
${PY_LIB_DIR}/pre-release:\
${PY_LIB_DIR}/amr_coref:${PYTHONPATH}"
    cd ${ENV_ROOT_DIR}
}

function print_python_env() {
    python --version
    python -c 'import sys ; print(sys.executable)'
}
