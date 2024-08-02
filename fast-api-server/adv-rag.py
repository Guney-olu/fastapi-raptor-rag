from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import tqdm

# Set up HTTPS context for SentenceTransformer
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

model_st = SentenceTransformer('all-MiniLM-L6-v2')

def split_text(text, n):
    avg_len = len(text) // n
    splits = []
    for i in range(n):
        start_idx = i * avg_len
        end_idx = (i + 1) * avg_len if i != n - 1 else len(text)
        splits.append(text[start_idx:end_idx])
    return splits

# Function to load, split, and add metadata to PDF
def process_pdf(file_path, link):
    reader = PdfReader(file_path)
    full_text = " ".join([page.extract_text() for page in reader.pages])
    chunks = split_text(full_text, 3)
    docs_with_metadata = [[chunk, link] for chunk in chunks]
    return docs_with_metadata

# Paths and links for the PDFs
pdf_data = []
"""
Sample format
pdf_data = [
    ("/kaggle/input/mental-health-data/data 2/GeneralisedanxietydisorderGAD.pdf", "https://www.betterhealth.vic.gov.au/health/healthyliving/Generalised-anxiety-disorder"),
]
"""
# Process each PDF and concatenate the results
final_chunk = []
for file_path, link in pdf_data:
    docs_with_metadata = process_pdf(file_path, link)
    final_chunk.extend(docs_with_metadata)


def prepare_data(parsed_data):
    data = []
    for t, l in enumerate(tqdm.tqdm(parsed_data, desc="Preparing data")):
        combined_text = l[0]
        embedding = model_st.encode(combined_text).tolist() 
        data.append({
            "id": t,
            "vector": embedding,
            "text": combined_text,
            "link": l[1]
        })
    return data


data = prepare_data(final_chunk)


milvus_client = MilvusClient(uri="/kaggle/working/bookfusion.db")

collection_name = "BF_collection"
if milvus_client.has_collection(collection_name):
     milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
     collection_name=collection_name,
     dimension=len(data[0]['vector']),
     metric_type="IP",  # Inner product distance
     consistency_level="Strong",  # Strong consistency level
 )

# # Insert data into Milvus
milvus_client.insert(collection_name=collection_name, data=data)

print("Data insertion completed.")