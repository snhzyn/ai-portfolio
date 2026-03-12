import uuid

from fastapi import APIRouter
from app.schemas.api_models import ContentGenerateRequest, ContentGenerateResponse
from app.graph.workflow import build_graph

router = APIRouter()
graph = build_graph()


@router.post("/content/generate", response_model=ContentGenerateResponse)
def generate_content(request: ContentGenerateRequest):
    request_id = str(uuid.uuid4())

    initial_state = {
        "request_id": request_id,
        "request": request.model_dump(),
    }

    result = graph.invoke(initial_state)

    return ContentGenerateResponse(
        request_id=request_id,
        result=result.get("final_json", {}),
    )