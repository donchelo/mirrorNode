import requests
import numpy as np
import torch
from PIL import Image
import io
import tempfile
import os
import json
import time
from typing import Optional, Tuple, Dict, Any

class YourMirrorVirtualTryOn:
    """
    YourMirror.io Virtual Try-On Node for ComfyUI
    Integrates with YourMirror.io API for virtual garment try-on functionality
    """
    
    def __init__(self):
        self.api_base_url = "https://apiservice.yourmirror.io"
        self.config_file = os.path.join(os.path.dirname(__file__), "config.json")
        self.max_retries = 3
        self.timeout = 60
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "product_image": ("IMAGE",),
                "workflow_type": (["eyewear", "footwear", "dress", "bottom", "top"],),
                "quality": (["normal", "high"],),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Enter your YourMirror.io API key"
                }),
            },
            "optional": {
                "mask_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result_image",)
    FUNCTION = "generate_tryon"
    CATEGORY = "YourMirror"
    DESCRIPTION = "Virtual try-on using YourMirror.io API"
    
    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert ComfyUI tensor to PIL Image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension if present
        
        # Convert from (C, H, W) to (H, W, C) if needed
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute(1, 2, 0)
        
        # Ensure values are in [0, 255] range
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        # Convert to numpy and ensure uint8
        np_image = tensor.cpu().numpy().astype(np.uint8)
        
        # Handle different channel configurations
        if np_image.shape[-1] == 4:
            # RGBA
            return Image.fromarray(np_image, 'RGBA')
        elif np_image.shape[-1] == 3:
            # RGB
            return Image.fromarray(np_image, 'RGB')
        else:
            # Grayscale
            return Image.fromarray(np_image.squeeze(), 'L')
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI tensor"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        np_image = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(np_image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def save_temp_image(self, pil_image: Image.Image, format: str = "PNG") -> str:
        """Save PIL image to temporary file and return file path"""
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{format.lower()}", 
            prefix="yourmirror_"
        )
        
        # Convert to RGB if saving as JPEG
        if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P"):
            # Create white background for transparency
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        
        pil_image.save(temp_file.name, format=format, quality=95)
        temp_file.close()
        return temp_file.name
    
    def cleanup_temp_files(self, *file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {file_path}: {e}")
    
    def make_api_request(self, payload: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Make API request with retry logic"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'ComfyUI-MirrorNode/1.0'
            }
            
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # Handle different response scenarios
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    raise Exception(f"API Error: {result['error']}")
                return result
            elif response.status_code == 429:  # Rate limit
                if retry_count < self.max_retries:
                    wait_time = (2 ** retry_count) * 5  # Exponential backoff
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    return self.make_api_request(payload, retry_count + 1)
                else:
                    raise Exception("Rate limit exceeded. Please wait before making more requests.")
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                print(f"Request timeout. Retrying... ({retry_count + 1}/{self.max_retries})")
                return self.make_api_request(payload, retry_count + 1)
            else:
                raise Exception("Request timed out after multiple retries")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
    
    def download_result_image(self, image_url: str) -> Image.Image:
        """Download the result image from the provided URL"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            raise Exception(f"Failed to download result image: {str(e)}")
    
    def validate_inputs(self, base_image, product_image, workflow_type, quality, api_key, mask_image=None):
        """Validate all inputs before making API request"""
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required. Get your API key from yourmirror.io")
        
        if workflow_type not in ["eyewear", "footwear", "dress", "bottom", "top"]:
            raise ValueError(f"Invalid workflow type: {workflow_type}")
        
        if quality not in ["normal", "high"]:
            raise ValueError(f"Invalid quality setting: {quality}")
        
        if base_image is None:
            raise ValueError("Base image (person) is required")
        
        if product_image is None:
            raise ValueError("Product image (garment) is required")
    
    def generate_tryon(self, base_image, product_image, workflow_type, quality, api_key, mask_image=None):
        """Main function to generate virtual try-on"""
        temp_files = []
        
        try:
            # Validate inputs
            self.validate_inputs(base_image, product_image, workflow_type, quality, api_key, mask_image)
            
            print(f"Starting YourMirror.io virtual try-on with workflow: {workflow_type}, quality: {quality}")
            
            # Convert tensors to PIL images
            base_pil = self.tensor_to_pil(base_image)
            product_pil = self.tensor_to_pil(product_image)
            mask_pil = self.tensor_to_pil(mask_image) if mask_image is not None else None
            
            # Save images to temporary files
            base_path = self.save_temp_image(base_pil, "PNG")
            product_path = self.save_temp_image(product_pil, "PNG")
            temp_files.extend([base_path, product_path])
            
            mask_path = None
            if mask_pil:
                mask_path = self.save_temp_image(mask_pil, "PNG")
                temp_files.append(mask_path)
            
            # Prepare API payload
            payload = {
                "data": [
                    {"path": base_path, "meta": {"_type": "gradio.FileData"}},
                    {"path": product_path, "meta": {"_type": "gradio.FileData"}},
                    workflow_type,
                    {"path": mask_path, "meta": {"_type": "gradio.FileData"}} if mask_path else None,
                    quality,
                    api_key.strip()
                ]
            }
            
            # Make API request
            print("Sending request to YourMirror.io API...")
            result = self.make_api_request(payload)
            
            # Extract result image URL
            if 'data' not in result or not result['data']:
                raise Exception("No image data returned from API")
            
            image_url = result['data'][0]
            print(f"Downloading result from: {image_url}")
            
            # Download and convert result image
            result_pil = self.download_result_image(image_url)
            result_tensor = self.pil_to_tensor(result_pil)
            
            print("Virtual try-on completed successfully!")
            return (result_tensor,)
            
        except Exception as e:
            error_msg = str(e)
            print(f"YourMirror.io Error: {error_msg}")
            
            # Create error image with text
            error_image = Image.new('RGB', (512, 512), color=(220, 53, 69))  # Bootstrap danger red
            result_tensor = self.pil_to_tensor(error_image)
            
            # Re-raise the exception so ComfyUI shows the error
            raise Exception(f"MirrorNode Error: {error_msg}")
            
        finally:
            # Clean up temporary files
            self.cleanup_temp_files(*temp_files)