

from langchain.callbacks.base import BaseCallbackManager 
from callback import StreamingLLMCallbackHandler


from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.callbacks import get_openai_callback

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path

from ingest import verify_filename_before_ingestion, ingest_doc_to_local_vectstore,  pinecone_namespace_to_vectorestore

def make_simple_LLMCHAIN(chatbotPrompt,model, temperature,max_tokens, openai_apik):
 """
 The make_simple_LLMCHAIN function is a Python function that constructs a simplified language model chain for generating responses to user prompts. 
 It takes various parameters including the chatbot prompt, model name, temperature, maximum tokens, OpenAI API key, and optional WebSocket 
 for streaming responses.

Parameters
chatbotPrompt (str): The initial prompt to guide the chatbot's responses.
model (str): The name of the OpenAI language model to use for generating responses.
temperature (float): The temperature parameter controlling the randomness of the response generation process.
max_tokens (int): The maximum number of tokens to generate in a response.
openai_apik (str): The OpenAI API key for authentication.

Return Value
Returns a language model chain (LLMChain) object configured with the specified parameters for generating responses to user prompts.
 """
 
 
 try:
    stream_manager = BaseCallbackManager ([])
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=False,
        callback_manager=stream_manager,
        verbose=False,
        openai_api_key=openai_apik,
        max_tokens=max_tokens)
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(chatbotPrompt, input_variables=["input"])
    human_template = """This is the Question: {input}
            Here is a context to help you find information:
            {vectstore_relevantdocs}
            End of context
            """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template, input_variables=["vectstore_relevantdocs", "input"])
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt, verbose=False)
    return chain
 except:
     print("ERROR in make_simple_LLMCHAIN")


async def answer_one_question(vectstore: VectorStore,input,
                              model,
                              questionAnsweringTemperature, 
                              similarSourceDocuments, maxTokens, 
                              openai_apik, chatbotPrompt)->str: #input is the question
 """
 The answer_one_question function is an asynchronous Python function responsible for generating an answer to a single user question 
 within a conversation. It leverages relevant documents from  the vector store to provide a contextually accurate response. 
 The function takes several parameters including the vector store, 
 user input (question), GPT model, temperature settings, and more.

Parameters
vectstore (VectorStore): A vector store object used for similarity search based on metadata attributes.
input (str): The user's question for which an answer is to be generated.
model (str): The name of the GPT model to be used for generating answers.
questionAnsweringTemperature (float): Temperature setting for generating the answer to the user question.
similarSourceDocuments (int): The number of similar source documents to retrieve from  the vector store.
maxTokens (int): The maximum number of tokens allowed for the generated response.
openai_apik (str): The OpenAI API key for using the OpenAI API.
chatbotPrompt (str): The chatbot prompt template for generating answers.
chunkSize (int): The size of the text chunks for text processing.
chunkOverlap (int): The overlap between text chunks for text processing.

Return Value
Returns a tuple containing:
resp (str): The generated response to the user's question.
relevantdocs (List[Document]): A list of relevant documents from the vector store.
 """
 try:

    standalone_question=input
   
    vectstore_relevantdocs = vectstore.similarity_search(query=standalone_question, k=similarSourceDocuments)

    chain=make_simple_LLMCHAIN(chatbotPrompt= chatbotPrompt,model= model,openai_apik= openai_apik,
                               temperature=questionAnsweringTemperature,max_tokens=maxTokens)
    
    with get_openai_callback() as cb:
            resp=chain.run(vectstore_relevantdocs=vectstore_relevantdocs, chat_relevantdocs=vectstore_relevantdocs, input=standalone_question)
            tokens=cb.total_tokens
            cost=cb.total_cost
            cost_message = "Cost: {} tokens spent, cost= ${}".format(tokens, cost)
    return resp, vectstore_relevantdocs, cost_message
 except Exception as e:
    print("ERROR in answer_one_question")  
        

def chat_local(docName:str, openaik:str, chatbotPrompt:str,
                    question:str, model:str, 
                   questionAnsweringTemperature, 
                   similarSourceDocuments, maxTokens, chunkSize, chunkOverlap
                   ):
 """
 The chat_api function is an asynchronous Python function that serves as an API for conducting a chat with the chatbot. 
 It takes user input in the form of a question to provide a contextually relevant response. 
 The function integrates the vector store and OpenAI GPT models for generating responses.

Parameters
openaik (str): The OpenAI API key for using the OpenAI API.
chatbotPrompt (str): The chatbot prompt template for generating answers.
question (str): The user's question for which an answer is to be generated.
model (str): The name of the GPT model to be used for generating answers.
questionAnsweringTemperature (float): Temperature setting for generating the answer to the user question.
similarSourceDocuments (int): The number of similar source documents to retrieve from  the vector store.
maxTokens (int): The maximum number of tokens allowed for the generated response.
chunkSize (int): The size of the text chunks for text processing.
chunkOverlap (int): The overlap between text chunks for text processing.

Return Value
Returns the generated response as a string, the vectstore_relevantdocs and cost_message
 """
 if not  verify_filename_before_ingestion(doc=docName):
    print("ERROR in verify filename_before_ingestion. "+ docName+" may not be of supported format.")
    return None
 try:
    vectstore=ingest_doc_to_local_vectstore(doc=docName, chunkSize=chunkSize, chunkOverlap=chunkOverlap, open_ai_key=openaik)
    resp, vectstore_relevantdocs, cost_message= answer_one_question(
                                     vectstore=vectstore,input=question,model=model,
                                     questionAnsweringTemperature=questionAnsweringTemperature,
                                     similarSourceDocuments=similarSourceDocuments,
                                     maxTokens=maxTokens, openai_apik=openaik,chatbotPrompt=chatbotPrompt)
    return resp, vectstore_relevantdocs, cost_message
 except Exception as e:
    print("ERROR in chat_api:")  
    print(e)
    
  
def loop_chat_local(openaik:str,
                    model:str, 
                   questionAnsweringTemperature, 
                   similarSourceDocuments, maxTokens
                   ):
 try:
    stream_manager = BaseCallbackManager ([])
    if not Path("vectorstore.pkl").exists():
        print("vectorstore.pkl does not exist, please run Ingest first")
        return None
    vectstore:VectorStore
    with open("vectorstore.pkl", "rb") as f:
        vectstore = pickle.load(f)
    qa_chain = ConversationalRetrievalChain.from_llm(
      ChatOpenAI(model=model,
        temperature=questionAnsweringTemperature,
        streaming=False,
        callback_manager=stream_manager,
        verbose=False,
        openai_api_key=openaik,
        max_tokens=maxTokens),
      vectstore.as_retriever(search_kwargs={'k': similarSourceDocuments}),
      return_source_documents=True,
      verbose=False)
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    chat_history = []
    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to Aabco Chatbot. You are now ready to start interacting with Jenny')
    print('---------------------------------------------------------------------------------')
    while True:
      query = input(f"{green}Question: ")
      if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting...')
        break
      if query == '':
        continue
      with get_openai_callback() as cb:
         result = qa_chain({"question": query, "chat_history": chat_history})
         tokens=cb.total_tokens
         cost=cb.total_cost
         cost_message = "Cost: {} tokens spent, cost= ${}".format(tokens, cost)
         print(f"{green}Source documents: \n")
         i=0
         for d in result["source_documents"]:
            i=i+1
            print("Chunk "+str(i)+": ")
            print(d.page_content)
            print()
         print()
         print(f"{white}Answer: " + result["answer"])
         print(cost_message)
         print()
         chat_history.append((query, result["answer"]))
    return chat_history
 except Exception as e:
    print("ERROR in loop_chat_local:")  
    print(e)
    
def loop_chat_pinecone(indexname:str, namespace:str, openaik:str,
                    model:str, 
                   questionAnsweringTemperature, 
                   similarSourceDocuments, maxTokens, pineconekey, pineconeenv, chat_history
                   ):
 try:
    stream_manager = BaseCallbackManager ([])
    vectstore:VectorStore
    vectstore=pinecone_namespace_to_vectorestore(pinecone_apik=pineconekey,open_apik=openaik,index_name=indexname,pinecone_env=pineconeenv, ns=namespace)
    if vectstore==None:
      print("Inexistent Pinecone index name or namespace")
      return None
    qa_chain = ConversationalRetrievalChain.from_llm(
      ChatOpenAI(model=model,
        temperature=questionAnsweringTemperature,
        streaming=False,
        callback_manager=stream_manager,
        verbose=False,
        openai_api_key=openaik,
        max_tokens=maxTokens),
      vectstore.as_retriever(search_kwargs={'k': similarSourceDocuments}),
      return_source_documents=True,
      verbose=False)
    yellow = "\033[0;33m"
    green = "\033[0;32m"
    white = "\033[0;39m"

    print(f"{yellow}---------------------------------------------------------------------------------")
    print('Welcome to Aabco Chatbot. You are now ready to start interacting with Jenny')
    print('---------------------------------------------------------------------------------')
    while True:
      query = input(f"{green}Question: ")
      if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting...')
        break
      if query == '':
        continue
      with get_openai_callback() as cb:
         result = qa_chain({"question": query, "chat_history": chat_history})
         tokens=cb.total_tokens
         cost=cb.total_cost
         cost_message = "Cost: {} tokens spent, cost= ${}".format(tokens, cost)
         # print(f"{green}Source documents: \n")
         # i=0
         # for d in result["source_documents"]:
         #    i=i+1
         #    print("Chunk "+str(i)+": ")
         #    print(d.page_content)
         #    print()
         print()
         print(f"{white}Answer: " + result["answer"])
         print(cost_message)
         print()
         chat_history.append((query, result["answer"]))
    return chat_history
 except Exception as e:
    print("ERROR in loop_chat_pinecone:")  
    print(e)
    
def answer_one_session_question(query, pineconekey,openaik,indexname,pineconeenv,pineconenamespace,model,questionAnsweringTemperature,maxTokens,
                                similarSourceDocuments, chat_history):
    stream_manager = BaseCallbackManager ([])
    
    vectstore:VectorStore
    vectstore=pinecone_namespace_to_vectorestore(pinecone_apik=pineconekey,open_apik=openaik,index_name=indexname,pinecone_env=pineconeenv, ns=pineconenamespace)
    if vectstore==None:
      print("Inexistent Pinecone index name or namespace")
      return None
    qa_chain = ConversationalRetrievalChain.from_llm(
      ChatOpenAI(model=model,
        temperature=questionAnsweringTemperature,
        streaming=False,
        callback_manager=stream_manager,
        verbose=False,
        openai_api_key=openaik,
        max_tokens=maxTokens
      ),
        vectstore.as_retriever(search_kwargs={'k': similarSourceDocuments}),
        return_source_documents=False, 
        verbose=False
    )
  
    result = qa_chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"],chat_history


async def answer_one_session_question_streaming(query, pineconekey,openaik,indexname,pineconeenv,pineconenamespace,model,questionAnsweringTemperature,maxTokens,
                                similarSourceDocuments, chat_history, websocket):
    stream_handler = StreamingLLMCallbackHandler(websocket)
    stream_manager = BaseCallbackManager([stream_handler])
    vectstore:VectorStore
    vectstore=pinecone_namespace_to_vectorestore(pinecone_apik=pineconekey,open_apik=openaik,index_name=indexname,pinecone_env=pineconeenv, ns=pineconenamespace)
    if vectstore==None:
      print("Inexistent Pinecone index name or namespace")
      return None
    qa_chain = ConversationalRetrievalChain.from_llm(
      ChatOpenAI(model=model,
        temperature=questionAnsweringTemperature,
        streaming=True,
        callback_manager=stream_manager,
        verbose=False,
        openai_api_key=openaik,
        max_tokens=maxTokens
      ),
        vectstore.as_retriever(search_kwargs={'k': similarSourceDocuments}),
        return_source_documents=False, 
        verbose=False
    )
    result=await qa_chain.arun(input=query,chat_history=chat_history )
    chat_history.append((query, result))
    return result,chat_history

#TESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSST



#"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"
# # "gpt-3.5-turbo"  "gpt-4-0314"
# chatbotPrompt="""
#    You are a conversational AI Sales assistant called Jenny; you only answer questions based on the documents provided.
#    You are nice and respectfull. If you don't know the answer, do not create answers, just say that you don't know.
# """
# question="Has Aabco provided client Winner with the plans requested last monday regarding their last building at Dubai?"
#question="Did Winner send an acknowledgement after Aabco has sent them the plans regarding their last building at Dubai?"
# resp, vectstore_relevantdocs, cost_message=chat_api(docName="C:/Users/Khattar/source/repos/Aabco_work_history/testfile.txt", 
#            openaik="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",
#            chatbotPrompt=chatbotPrompt,
#            question=question,
#            model="gpt-4-0314",
#            questionAnsweringTemperature=0.9,
#            similarSourceDocuments=3,
#            maxTokens=3000,
#            chunkSize=500,
#            chunkOverlap=50
#            )
# print("Relevant Docs:")
# i=0
# for d in vectstore_relevantdocs:
#       i=i+1
#       print("Chunk "+str(i)+": "+d.page_content)
#       print()
# print()
# print("answer: "+resp)
# print()
# print(cost_message)
# print()




