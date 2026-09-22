"""
opencode_bridge.py - FastAPI HTTP Bridge for Mycelium Swarm OpenCode Integration.
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging

from orchestrator import get_orchestrator

# Initialize logger
logger = logging.getLogger("main")

app = FastAPI(
    title="Mycelium Sigma HTTP Bridge",
    description="HTTP API bridge for integration with OpenCode and local user interfaces.",
    version="8.9.2"
)

class SwarmRequest(BaseModel):
    task: str
    type: Optional[str] = "general"

class SwarmResponse(BaseModel):
    answer: str
    e_total: float
    delta_I: float
    step_time: float
    models: List[str]

class StatusResponse(BaseModel):
    step: int
    I: float
    lambda_L: float
    k_act: int
    regime: str
    alive_agents: int
    tokens_used: int

@app.post("/run", response_model=SwarmResponse)
async def run_swarm_step(request: SwarmRequest):
    """
    Executes a single step in the Mycelium swarm.
    Does not block the uvicorn event loop.
    """
    logger.info(f"HTTP Bridge received task: {request.task[:100]}... (Type: {request.type})")
    try:
        orch = get_orchestrator()
        # run_step is already a native async function!
        result = await orch.run_step(request.task, request.type)
        return SwarmResponse(
            answer=result['answer'],
            e_total=round(result['e_total'], 4),
            delta_I=round(result.get('delta_I', 0.0), 6),
            step_time=round(result['step_time'], 2),
            models=result['models']
        )
    except Exception as e:
        logger.error(f"Error executing swarm step: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_swarm_status():
    """
    Returns current thermodynamic and swarm state metrics.
    """
    try:
        orch = get_orchestrator()
        return StatusResponse(
            step=orch.step,
            I=round(orch.memory.I, 4),
            lambda_L=round(orch.chaos.current_lambda, 4),
            k_act=orch.chaos.current_k_act,
            regime=orch.chaos._regime(orch.chaos.current_lambda),
            alive_agents=sum(1 for a in orch.bandit.agents.values() if a.alive),
            tokens_used=orch.tokens_used
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Mycelium Sigma HTTP Bridge on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
