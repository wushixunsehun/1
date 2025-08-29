from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from hpc_rca import locate_root_causes

app = FastAPI(title="HPC Root Cause Analysis API")

class RCARequest(BaseModel):
    folder_path: str
    job_id: str
    failure_start: int
    failure_end: int
    failure_type: str
    root_node: str
    golden_metrics: Optional[List[str]] = None
    top_k: int = 5

@app.post("/locate_root_cause")
def locate_root_cause(request: RCARequest):
    result = locate_root_causes(
        folder_path=request.folder_path,
        job_id=request.job_id,
        failure_start=request.failure_start,
        failure_end=request.failure_end,
        failure_type=request.failure_type,
        root_node=request.root_node,
        golden_metrics=request.golden_metrics,
        top_k=request.top_k,
    )
    return result

if __name__ == "__main__":
    uvicorn.run("rca_api:app", host="0.0.0.0", port=5410, reload=True)
