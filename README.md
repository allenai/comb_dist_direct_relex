# Combining Distant and Direct Supervision for Neural Relation Extraction
This is code for our NAACL 2019 paper on combining distant and direct supervision to improve relation extraction. The code is implemented using PyTorch and AllenNLP. 

### Running The Code
After cloning this repository, follow the steps below for training and prediction.

1) Install requirements (mainly AllenNLP)
```
pip install -r requirements.txt
```

2) Use the following scrip to start training. Make sure to check and edit the parameters in the training script. The default parameters will train the model for **one epoch** on a **subset of the dataset**.
```
./scripts/train.sh serialization_dir
```

3) To run the trained model for prediction,
```
allennlp predict serialization_dir/model.tar.gz tests/fixtures/data.txt --include-package relex --cuda-device 0 --batch-size 32 --use-dataset-reader --predictor relex --output-file predictions.json
```
`predictions.json` contains model predictions for the examples provided in `tests/fixtures/data.txt`


### Citation
```
@inproceedings{Beltagy2019Comb,
  title={Combining Distant and Direct Supervision for Neural Relation Extraction},
  author={Iz Beltagy and Kyle Lo and Waleed Ammar},
  year={2019},
  booktitle={NAACL}
}
```
