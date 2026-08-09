from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    agent: str
    response: str
    elapsed: float
    is_error: bool
    # Populated only for the "image" intent on success — parsed from
    # ImageAgent's success message at the API boundary (see
    # api/routes/chat.py::_extract_image_url) so agents don't need to
    # return structured data just for this one frontend affordance.
    image_url: str | None = None
    # Running totals for this session (see core/cost_tracking.py). Doesn't
    # cover RAG or image-generation cost — see that module's docstring.
    session_cost_usd: float
    session_tokens: int
