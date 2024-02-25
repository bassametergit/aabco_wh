"""Callback handlers used in the app."""
from typing import Any, Dict, List
from schemas import ChatResponse


from langchain.callbacks.base import AsyncCallbackHandler

    
class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """Callback handler for question generation."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        pass
    
class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""
    sendit:bool
    def __init__(self, websocket):
        self.websocket = websocket
        self.sendit=False
    """
    we don't stream the heading Router Chain information which is a json: 
    { 
        "destination":"selected chain name", 
        "next_inputs":"enhanced question to fit one of the chains prompts"
    }
    so we wait the final "}" token to allow streaming answer (set 'sendit' to True)
    """
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        try:
            await self.websocket.send_json(resp.dict())
            
        except Exception as e:
            pass       

