# Combining Distant and Direct Supervision for Neural Relation Extraction

pip install -r requirements.txt

./scripts/train.sh  tmp/output..

allennlp predict tmp/output92/model.tar.gz  tests/fixtures/data.txt --include-package relex --cuda-device 0  --batch-size 32 --use-dataset-reader --output-file tmp/output.json --predictor relex

