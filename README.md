# HeartFailureML

Installation:

```
python -m venv env
source env/bin/activate
pip install -e .
python src/train_pipeline.py configs/train_config.yml
```

Docker
```
python setup.py sdist
docker build -t  heart_failure:v0 .
docker run --name heart_failure heart_failure:v0 
```


