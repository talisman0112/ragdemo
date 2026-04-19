from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
md5_path = "./md5.txt"
#chroma_db
collection_name="knowledge_base"
persist_directory="./chroma_db"
#embedding_function
embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
#spliter
spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", "","?",".","!","@","#","$","%","^","&","*","(",")","[","]","{","}","<",">","/","\\","|","`","~","-","_","+","=",":","\;","\"","\'","\""],length_function=len)
start_chunk_size=1000