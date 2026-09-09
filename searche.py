from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
import os
from dotenv import load_dotenv
load_dotenv()
def search(question,groq_api_key):
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=100000)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
    wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100000)
    wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)
    search=DuckDuckGoSearchRun(name="Search")
    llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant",streaming=True)
    tools=[arxiv,search,wiki]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parse_errors=True)
    response=search_agent.run(question)
    return response
