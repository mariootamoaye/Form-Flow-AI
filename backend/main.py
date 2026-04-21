"""Form-Flow-AI Backend Entry Point.

FastAPI application that powers the conversational form-filling assistant,
providing REST and WebSocket endpoints for real-time AI-driven form interactions.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import conversation, forms, health
from app.core.config import settings
from app.core.logging import setup_logging

# Configure structured logging before anything else
setup_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("Starting Form-Flow-AI backend", extra={"version": settings.APP_VERSION})
    # Initialize services (RAG index, model connections, etc.)
    from app.services.rag_service import RAGService
    from app.services.gemini_service import GeminiService

    app.state.rag_service = RAGService()
    app.state.gemini_service = GeminiService(api_key=settings.GEMINI_API_KEY)

    await app.state.rag_service.initialize()
    logger.info("All services initialized successfully")

    yield

    # Graceful shutdown
    logger.info("Shutting down Form-Flow-AI backend")
    await app.state.rag_service.close()


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title="Form-Flow-AI API",
        description="Conversational AI assistant for intelligent form filling",
        version=settings.APP_VERSION,
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        lifespan=lifespan,
    )

    # CORS — allow configured origins (frontend dev server, production domain)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(forms.router, prefix="/api/v1/forms", tags=["forms"])
    app.include_router(conversation.router, prefix="/api/v1/conversation", tags=["conversation"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        logger.exception("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
