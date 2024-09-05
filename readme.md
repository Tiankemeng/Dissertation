## Research on Meta-Learning Algorithm Based on Bert on Question Answering Dataset

### Datasets
This project uses two data sets, personal self-built data (data/ipqa) and wikiQA (data/wikiQA) data sets, to evaluate the meta-learning algorithm.

### MOdel
This project uses the Bert pre-trained model (pretrained_models/bert-base-uncased) as the basic model for algorithm exploration experiments.

### Algorithm
The meta-learning algorithms evaluated in this project include maml, prototype network (pn) and reptile algorithm. The algorithm code is in the engine_train.py file.

### Evaluation indicators
This project uses two common evaluation indicators, MAP and MRR, and the calculation code is in the metrics.py file.

### Code running
The code running command is in run.sh. Running a single-line command is to run the corresponding algorithm of the corresponding data set. Running run.sh directly can get the results of the three algorithms in 2 data sets and draw a comparison chart. The results are saved in results, and the comparison chart is in the results/lr_1e-5_ep50/summary folder.

### Code running command
`bash run.sh`
