from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from llama_cpp import Llama
import os
cwd = os.getcwd()

db_path  = cwd + "/bookfusion.db"

model_st = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

@app.get("/")
def public():
    html_content = """
<html>
<head>
    <title>THE API Endpoint</title>
    <style>
        body {
            text-align: center;
        }
        h1, p {
            font-size: 24px;
        }
        h1 {
            font-size: 36px;
        }
        a {
            font-size: 20px;
            color: blue;
            text-decoration: none; 
        }
        a:hover {
            text-decoration: underline; 
        }
    </style>
</head>
<body>
    <h1>go to /docs</h1>
    <p> two specific endpoints, "/rag" and "/classification" , each serving a distinct purpose</p>
</body>
</html>
    """
    return HTMLResponse(content=html_content)




def rag_call(question):
    milvus_client = MilvusClient(uri=db_path)
    collection_name = "BF_collection"
    query_res = milvus_client.query(
    collection_name=collection_name,
    filter="", 
    output_fields=["text"],
    limit=100  
    )
    all_texts = [entity['text'] for entity in query_res]
    bm25 = BM25Okapi([text.split() for text in all_texts])
    bm25_scores = bm25.get_scores(question.split())
    bm25_top_n_indices = bm25.get_top_n(question.split(), all_texts, n=10)
    candidate_embeddings = [model_st.encode(text).tolist() for text in bm25_top_n_indices]
    expanded_query_embeddings = [model_st.encode(term).tolist() for term in question]

    # Hybrid retrieval: re-rank candidates using DPR and Milvus
    search_results = []
    for embedding in expanded_query_embeddings:
        search_res = milvus_client.search(
            collection_name=collection_name,
            data=[embedding],
            limit=1,  # Return top 3 results
            search_params={"metric_type": "IP", "params": {}}, 
            output_fields=["text", "link"],
        )
        search_results.extend(search_res[0])

    retrieved_lines_with_distances = [
        {
            "text": res["entity"]["text"],
            "link": res["entity"]["link"],
            "distance": res["distance"]
        }
        for res in search_results
    ]


    unique_results = list({(result['text'], result['link'], result['distance']): result for result in retrieved_lines_with_distances}.values())
    sorted_results = sorted(unique_results, key=lambda x: x["distance"], reverse=True)
    top_results = sorted_results[:1]
    
    return top_results


# def llama_call(prompt,context):
#     llm = Llama(
#       model_path="/kaggle/working/llama-2-7b-32k-instruct.Q2_K.gguf",
#       chat_format="llama-2",
#       n_ctx=2048,
#     )
#     inp = f"context : {context}   and   question : {prompt}"
#     response =llm.create_chat_completion(
#       messages = [
#           {"role": "system", "content": "You are an an assistant which give answer to question based on context"},
#           {
#               "role": "user",
#               "content": inp
#           }
#         ],
#         max_tokens = 500
#     )
#     return response
    

@app.get("/rag")
def rag(prompt):
    results = rag_call(prompt)
    context = " ".join([result["text"] for result in results])
    title = " & ".join([result["link"] for result in results])  
    # response = llama_call(prompt,context)
    # output = response['choices'][0]['message']['content']
    return context,title
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)