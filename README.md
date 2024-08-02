# FAST API, RAG and ENDPOINTS. . .

```bash
git clone https://github.com/Guney-olu/fastapi-raptor-rag.git
python3 pip install -r requirements.txt
```

**Download stuff**
Classification model NEEDED
```bash
https://drive.google.com/file/d/1ep3RyEegw53X01SE1MAHx48-WzTsiY3A/view?usp=drive_link
```
GGUF LLM for rag chat
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q2_K.gguf
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

[file](fast-api-server/cls_training.py)

### Docker Build

```bash
docker build -t my-fastapi-app .
docker run -d -p 8000:8000 my-fastapi-app
```
