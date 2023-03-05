#!/usr/bin/env bash
. ./script/mecab_install.sh
cd /code
uvicorn app.main:app --host 0.0.0.0 --port 3000

