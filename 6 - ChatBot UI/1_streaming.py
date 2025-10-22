from utils import load_workflow
from langchain_core.messages import HumanMessage

workflow = load_workflow()

prompt = "hi write an essay of 200 words"

config = {"configurable": {"thread_id": "1"}}

# res_generator = workflow.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="messages")

for message_chunk, meta_data in workflow.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="messages"): # type: ignore
    
    if message_chunk.content: # type: ignore
        print(message_chunk.content, end="", flush=True) # type: ignore
