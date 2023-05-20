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
python eval_counterfactual.py -c training_logs/counterfactual_lungs_cgan-May-14-2023_09+38PM-f54b319/hparams.yaml -cp training_logs/counterfactual_lungs_cgan-May-14-2023_09+38PM-f54b319 -t 0.5
```

```
rsync /gpfs/space/home/dmytrosh/counterfactual-search/COVID-19_Radiography_Dataset_v2/ /tmp/
```
