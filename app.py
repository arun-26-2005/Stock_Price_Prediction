import gradio as gr
from backend.app import app as fastapi_app
import uvicorn

# Simple health check helper for Gradio Interface
def api_status(message):
    return f"StockAI Pro API Server is active. Echo: '{message}'"

# Create Gradio demo interface
demo = gr.Interface(
    fn=api_status,
    inputs=gr.Textbox(placeholder="Type message to test API..."),
    outputs="text",
    title="StockAI Pro API Server",
    description="Active Gradio mount hosting the FastAPI stock prediction backend endpoints."
)

# Mount Gradio onto our FastAPI application.
# This keeps our FastAPI server at the root "/" and exposes the Gradio GUI at "/gui"
app = gr.mount_gradio_app(fastapi_app, demo, path="/gui")

# Start Uvicorn ASGI server directly to keep the Hugging Face container alive
# listens on port 7860 (Hugging Face default port)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
