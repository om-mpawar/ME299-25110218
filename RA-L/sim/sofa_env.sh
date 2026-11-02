# Source this before running a scene:  export SOFA_ROOT=<SOFA 25.06 install>; source sim/sofa_env.sh
# SOFA 25.06 with SofaPython3, SoftRobots and STLIB (not on PyPI); its Python is 3.12, like the repository's .venv.
: "${SOFA_ROOT:?set SOFA_ROOT to the SOFA 25.06 installation}"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$SOFA_ROOT/plugins/SofaPython3/lib:$SOFA_ROOT/plugins/SoftRobots/lib/python3/site-packages:$SOFA_ROOT/plugins/STLIB/lib/python3/site-packages:$SOFA_ROOT/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
