from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from debugpy.common import messaging, sockets

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield

app = FastAPI(lifespan=lifespan)

debug_channel = None

class ConnectRequest(BaseModel):
    port: int = 5678
    host: str = "localhost"
    pid: Optional[int] = None

class BreakpointRequest(BaseModel):
    file: str
    line: int
    condition: Optional[str] = None
    hit_condition: Optional[str] = None

class EvaluateRequest(BaseModel):
    expression: str
    frame_id: Optional[int] = None
    context: str = "repl"  # "repl", "watch", or "hover"

class ConnectionStatus(BaseModel):
    status: str
    connected: bool
    session_id: Optional[str] = None


@app.post("/connect", response_model=ConnectionStatus)
async def connect(request: ConnectRequest):
    """Connect to a debug session"""
    global debug_channel
    try:
        ipv6 = request.host.count(":") > 1
        sock = sockets.create_client(ipv6)
        sock.connect((request.host, request.port))
        stream = messaging.JsonIOStream.from_socket(sock, "debugger")
        debug_channel = messaging.JsonMessageChannel(stream, handlers={})
        debug_channel.start()
        
        # Initialize the debug session
        debug_channel.request("initialize", {
            "clientID": "middleware",
            "clientName": "Debug Middleware",
            "adapterID": "python",
            "pathFormat": "path",
            "linesStartAt1": True,
            "columnsStartAt1": True,
            "supportsVariableType": True,
            "supportsVariablePaging": True,
            "supportsRunInTerminalRequest": True,
            "locale": "en-us"
        })
        
        debug_channel.request("attach", {
            "name": "Python Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": request.host,
                "port": request.port
            },
            "pathMappings": [{"localRoot": ".", "remoteRoot": "."}]  # Adjust as needed
        })
        
        debug_channel.request("configurationDone")
        
        return ConnectionStatus(
            status="connected",
            connected=True,
            session_id=str(id(debug_channel))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to connect: {str(e)}")

@app.get("/status", response_model=ConnectionStatus)
async def get_status():
    """Get current connection status"""
    return ConnectionStatus(
        status="connected" if debug_channel is not None else "disconnected",
        connected=debug_channel is not None,
        session_id=str(id(debug_channel)) if debug_channel else None
    )

@app.get("/variables/{frame_id}")
async def get_variables(frame_id: int):
    """Get variables for a specific frame"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        response = debug_channel.request('variables', {
            'variablesReference': frame_id
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get variables: {str(e)}")

@app.get("/stacktrace")
async def get_stacktrace(thread_id: int = 1, start_frame: int = 0, levels: int = 20):
    """Get current stack trace"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        response = debug_channel.request('stackTrace', {
            'threadId': thread_id,
            'startFrame': start_frame,
            'levels': levels
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stack trace: {str(e)}")

@app.post("/breakpoint")
async def set_breakpoint(request: BreakpointRequest):
    """Set a breakpoint"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        breakpoint_data = {'line': request.line}
        if request.condition:
            breakpoint_data['condition'] = request.condition
        if request.hit_condition:
            breakpoint_data['hitCondition'] = request.hit_condition
            
        response = debug_channel.request('setBreakpoints', {
            'source': {'path': request.file},
            'breakpoints': [breakpoint_data]
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set breakpoint: {str(e)}")

@app.delete("/breakpoint")
async def clear_breakpoints(file: str):
    """Clear all breakpoints in a file"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        response = debug_channel.request('setBreakpoints', {
            'source': {'path': file},
            'breakpoints': []
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear breakpoints: {str(e)}")

@app.post("/evaluate")
async def evaluate_expression(request: EvaluateRequest):
    """Evaluate an expression in the current context"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        response = debug_channel.request('evaluate', {
            'expression': request.expression,
            'frameId': request.frame_id,
            'context': request.context
        })
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate expression: {str(e)}")

@app.get("/threads")
async def get_threads():
    """Get all threads in the debug session"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        response = debug_channel.request('threads')
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get threads: {str(e)}")

@app.post("/continue")
async def continue_execution(thread_id: Optional[int] = None):
    """Continue execution"""
    try:
        args = {}
        if thread_id is not None:
            args['threadId'] = thread_id
            
        response = debug_channel.request('continue', args)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to continue: {str(e)}")

@app.post("/step")
async def step(step_type: str = "in", thread_id: Optional[int] = None):
    """Step through code (in/over/out)"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    if step_type not in ["in", "over", "out"]:
        raise HTTPException(status_code=400, detail="step_type must be 'in', 'over', or 'out'")
    
    command_map = {
        "in": "stepIn",
        "over": "next",
        "out": "stepOut"
    }
    
    try:
        args = {}
        if thread_id is not None:
            args['threadId'] = thread_id
            
        response = debug_channel.request(command_map[step_type], args)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to step: {str(e)}")

@app.post("/pause")
async def pause_execution(thread_id: Optional[int] = None):
    """Pause execution"""
    if not debug_channel:
        raise HTTPException(status_code=400, detail="Not connected to debug session")
    
    try:
        args = {}
        if thread_id is not None:
            args['threadId'] = thread_id
            
        response = debug_channel.request('pause', args)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, threads=1)
