# Extracting Parallel Sentences with Bidirectional Recurrent Neural Networks to Improve Machine Translation
A TensorFlow implementation of the bidirectional RNN model described in the paper [Extracting Parallel Sentences with Bidirectional Recurrent Neural Networks to Improve Machine Translation
](https://arxiv.org/abs/1806.05559) to extract parallel sentences from aligned comparable corpora.

## Required packages
* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **NumPy** ([instructions](https://www.scipy.org/install.html))
* **scikit-learn** ([instructions](http://scikit-learn.org/stable/install.html))

## Prepare the training data
We have provided a script to tokenize and clean your datasets using [Moses](https://github.com/moses-smt/mosesdecoder).
```
./scripts/preprocessing.sh ~/moses/mosesdecoder ../data/train en fr 3 80
mv ../data/train.clean.en ../data/train.en
mv ../data/train.clean.fr ../data/train.fr
```

## Training
Run the training script.
```
python train.py --source_train_path ../data/train.en --target_train_path ../data/train.fr --source_valid_path ../data/valid.en --target_valid_path ../data/valid.fr --checkpoint_dir ../tflogs
```
The models are written in `checkpoint_dir`.

## Testing
Run the evaluation script.
```
python eval.py --checkpoint_dir ../tflogs --source_test_path ../data/test.en --target_test_path ../data/test.fr --reference_test_path ../data/test.ref --source_vocab_path ../data/vocabulary.source --target_vocab_path ../data/vocabulary.target
```
The evaluation is done on the last model saved in `checkpoint_dir`.

## Extracting sentence pairs
Run the sentence extraction script.
```
python extract.py --checkpoint_dir ../tflogs --extract_dir ./samples --source_vocab_path ../data/vocabulary.source --target_vocab_path ../data/vocabulary.target --source_output_path ../data/extracted.source --target_output_path ../data/extracted.target --score_output_path ../data/extracted.score --source_language en --target_language fr --decision_threshold 0.99
```
