# FAST API, RAG and ENDPOINTS. . .

```bash
git clone https://github.com/Guney-olu/fastapi-raptor-rag.git
python3 pip install -r requirements.txt
```

**Go to main.py before and chain accordinly to your gguf model path and setting**
```bash
cd fast-api-server
python3 main.py
```
*NOW GO TO*
```bash
https://127.0.0.1:8000/docs
```

### Rag chat DB

The fast-api-server/bookfusion.db is feed with few data (articles) to give the idea how a large scale application can be made and a new age search engine

Check file to know more
```bash
fast-api-server/adv-rag.py
```
### Classification Model

A classifier model trained using old school tensorflow on data https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

[text](fast-api-server/cls_training.py)

