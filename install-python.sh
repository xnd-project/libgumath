#!/usr/bin/env bash

set -e

if [[ -d ndtypes ]] ; then
  cd ndtypes
  python3 setup.py install --local=$PWD/../python
else
  echo "Did you do git submodule init?"
  exit 1
fi
cd ..

if [[ -d xnd ]] ; then
  cd xnd
  python3 setup.py install --local=$PWD/../python
else
  echo "Did you do git submodule init?"
  exit 1
fi
cd ..

python3 setup.py build -j$(getconf _NPROCESSORS_ONLN)

