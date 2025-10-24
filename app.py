"""
Main application file for the Floor Plan Agent API
Modularized version of the original application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import configuration and initialize settings
from modules.config import settings

# Import all routers
from modules.auth import auth_router
from modules.pdf_processing import pdf_router
from modules.projects import project_router
from modules.api import agent_router, general_router
from modules.api.session_endpoints import router as session_router

from modules.admin.endpoints import router as admin_router
from modules.subscription.endpoints import router as subscription_router
from modules.storage.endpoints import router as storage_router
from modules.feedback.endpoints import router as feedback_router
from modules.profile.endpoints import router as profile_router
from modules.notifications import notifications_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    # Validate settings
    settings.validate()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="AI-powered floor plan annotation and document analysis system with unified workflows"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://esticore.vercel.app","https://ecadocai.netlify.app", "https://app.ecadoc.ai"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(general_router)  # General endpoints (root, health, download)
    app.include_router(auth_router)     # Authentication endpoints
    app.include_router(pdf_router)      # Document processing endpoints
    app.include_router(project_router)  # Project management endpoints
    app.include_router(agent_router)    # Agent workflow endpoints
    app.include_router(session_router)  # Session management endpoints
    
    app.include_router(admin_router)
    app.include_router(subscription_router)
    app.include_router(storage_router)
    app.include_router(feedback_router)
    app.include_router(profile_router)
    app.include_router(notifications_router)

    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
