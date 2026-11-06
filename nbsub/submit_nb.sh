#!/bin/sh
# usage: submit_nb.sh <csv> "<message>"   (only run with the user's go-ahead)
set -e; cd "$(dirname "$0")"
cp "$1" data/submission.csv
kaggle datasets metadata samanyu1808/sis-final-packings -p data --update || true
kaggle datasets version -p data -m "$2" --dir-mode zip
until [ "$(kaggle datasets status samanyu1808/sis-final-packings 2>/dev/null | tail -1)" = ready ]; do sleep 5; done
V=$(kaggle kernels push -p kernel | grep -o "version [0-9]*" | cut -d" " -f2)
sleep 20; until kaggle kernels status samanyu1808/sis-submit 2>&1 | grep -qiE "complete|error"; do sleep 10; done
rm -rf out; kaggle kernels output samanyu1808/sis-submit -p out
cmp out/submission.csv "$1"
kaggle competitions submit squares-in-squares -k samanyu1808/sis-submit -f submission.csv -v "$V" -m "$2"
