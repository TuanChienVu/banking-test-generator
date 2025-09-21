#!/usr/bin/env python3
"""
FastAPI Server for Banking Test Case Generator
Serve model qua REST API cho web frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

# Add demo folder vào path để import model interface
sys.path.append(str(Path(__file__).parent / "demo"))

# Import model interface (sử dụng enhanced version cho output đẹp)
from model_interface_enhanced import EnhancedModelInterface

# Khởi tạo FastAPI app
app = FastAPI(
    title="Banking Test Case Generator API",
    description="Generate test cases for mobile banking using YOUR fine-tuned AI model",
    version="1.0.0"
)

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (lazy loading)
model_interface = None

# Request/Response models
class GenerateRequest(BaseModel):
    user_story: str
    test_type: str = "functional"
    max_length: int = 150
    use_template: bool = True  # Dùng template cho output nhất quán

class GenerateResponse(BaseModel):
    success: bool
    test_case: str  # HTML formatted test case
    metadata: Optional[dict] = None
    error: Optional[str] = None

def get_model():
    """Lazy load model - chỉ load khi cần"""
    global model_interface
    if model_interface is None:
        print("🔄 Loading model for the first time...")
        model_interface = EnhancedModelInterface()
        print("✅ Model loaded successfully!")
    return model_interface

@app.get("/")
async def root():
    """Root endpoint - kiểm tra API hoạt động"""
    return {
        "message": "Banking Test Case Generator API",
        "status": "running",
        "model": "Your Fine-tuned Model (8 epochs, 8000+ samples)",
        "endpoints": {
            "generate": "/api/generate",
            "health": "/health",
            "model_info": "/api/model_info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_interface is not None}

@app.get("/api/model_info")
async def get_model_info():
    """Lấy thông tin về model"""
    try:
        model = get_model()
        return {
            "success": True,
            "info": model.get_model_info(),
            "capabilities": {
                "test_types": ["functional", "security", "performance", "compliance"],
                "max_length_range": [50, 200],
                "features": [
                    "Login & Authentication",
                    "Money Transfer", 
                    "Balance Inquiry",
                    "Bill Payment",
                    "Card Management"
                ]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_test_case(request: GenerateRequest):
    """
    Generate test case từ user story
    
    Parameters:
    - user_story: Câu chuyện người dùng (As a... I want to...)
    - test_type: Loại test (functional/security/performance/compliance)
    - max_length: Độ dài tối đa của output
    - use_template: Dùng template (nhanh) hay model generation (chậm hơn)
    """
    try:
        # Validate input
        if not request.user_story or len(request.user_story.strip()) < 10:
            raise HTTPException(status_code=400, detail="User story is too short")
        
        if request.test_type not in ["functional", "security", "performance", "compliance"]:
            raise HTTPException(status_code=400, detail="Invalid test type")
        
        # Get model và generate
        model = get_model()
        result = model.generate_test_case(
            user_story=request.user_story,
            test_type=request.test_type,
            max_length=request.max_length,
            use_template=request.use_template
        )
        
        # Return response
        if result['success']:
            return GenerateResponse(
                success=True,
                test_case=result['test_case'],
                metadata=result.get('metadata', {})
            )
        else:
            return GenerateResponse(
                success=False,
                test_case="",
                error=result.get('error', 'Unknown error')
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error in generate_test_case: {e}")
        return GenerateResponse(
            success=False,
            test_case="",
            error=str(e)
        )

@app.post("/api/generate_batch")
async def generate_batch(test_cases: list[GenerateRequest]):
    """Generate nhiều test cases cùng lúc"""
    results = []
    model = get_model()
    
    for request in test_cases:
        try:
            result = model.generate_test_case(
                user_story=request.user_story,
                test_type=request.test_type,
                max_length=request.max_length,
                use_template=request.use_template
            )
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "user_story": request.user_story
            })
    
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Banking Test Case Generator API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
