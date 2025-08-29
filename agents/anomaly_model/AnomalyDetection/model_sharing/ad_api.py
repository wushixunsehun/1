from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from detecting import run_detect

app = FastAPI(title="HPC Time Series Anomaly Detection API")

class ADRequest(BaseModel):
    folder_path: str


@app.post("/time_series_ad")
def time_series_ad(request: ADRequest):
    node_df = pd.read_csv(request.folder_path, index_col=0, header=0)
    result = run_detect(node_df)
    return result

if __name__ == "__main__":
    uvicorn.run("ad_api:app", host="0.0.0.0", port=5411, reload=True)
