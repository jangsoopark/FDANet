#!/bin/bash

project_root="$(dirname ${PWD})"
pushd ${project_root}/notebook

jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

popd