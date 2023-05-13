## Setting up the project
```
pip install torch==2.0.0+cu117 torchvision --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/ufoym/imbalanced-dataset-sampler.git
pip install -e .
```

Relevant datasets:
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?datasetId=576013&sortBy=voteCount
(masks + classes)

https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
(only classes)


https://www.kaggle.com/code/arunrk7/covid-19-detection-pytorch-tutorial


```
python eval_counterfactual.py -c configs/counterfactual_lungs_cgan.yaml -cp training_logs/counterfactual_lungs_cgan-May-06-2023_07+11PM-d6a65d1 -t 0.5
```
