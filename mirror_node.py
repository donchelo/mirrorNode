import requests
import numpy as np
import torch
from PIL import Image
import io
import tempfile
import os
import json
import time
import base64
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
        self.log_messages = []  # Store log messages for output
        
    def log_debug(self, message: str):
        """Simple debug logging"""
        log_msg = f"[MirrorNode DEBUG] {message}"
        print(log_msg)
        self.log_messages.append(log_msg)
        
    def log_error(self, message: str):
        """Simple error logging"""
        log_msg = f"[MirrorNode ERROR] {message}"
        print(log_msg)
        self.log_messages.append(log_msg)
        
    def log_info(self, message: str):
        """Info logging"""
        log_msg = f"[MirrorNode INFO] {message}"
        print(log_msg)
        self.log_messages.append(log_msg)
    
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
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("result_image", "debug_logs")
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
        
        # Debug: Log file path and check if file exists
        self.log_debug(f"Saved temp image: {temp_file.name}")
        self.log_debug(f"File exists: {os.path.exists(temp_file.name)}")
        self.log_debug(f"File size: {os.path.getsize(temp_file.name)} bytes")
        
        return temp_file.name

    def save_image_to_gradio_temp(self, pil_image: Image.Image, format: str = "PNG") -> str:
        """Save image to a location that Gradio can serve as a public URL"""
        # Create a unique filename
        import uuid
        filename = f"yourmirror_{uuid.uuid4().hex}.{format.lower()}"
        
        # Save to a directory that Gradio can serve
        # This is a workaround - in a real implementation, you'd need to save to Gradio's temp directory
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, filename)
        
        # Convert to RGB if saving as JPEG
        if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P"):
            # Create white background for transparency
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        
        pil_image.save(file_path, format=format, quality=95)
        
        # Debug: Log file path and check if file exists
        self.log_debug(f"Saved image to Gradio temp: {file_path}")
        self.log_debug(f"File exists: {os.path.exists(file_path)}")
        self.log_debug(f"File size: {os.path.getsize(file_path)} bytes")
        
        return file_path

    def create_data_uri(self, pil_image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL image to data URI"""
        import base64
        import io
        
        # Convert image to bytes
        img_buffer = io.BytesIO()
        
        # Convert to RGB if saving as JPEG
        if format.upper() == "JPEG" and pil_image.mode in ("RGBA", "P"):
            # Create white background for transparency
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            if pil_image.mode == 'P':
                pil_image = pil_image.convert('RGBA')
            background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
            pil_image = background
        
        pil_image.save(img_buffer, format=format, quality=95)
        img_data = img_buffer.getvalue()
        
        # Convert to base64
        base64_data = base64.b64encode(img_data).decode('utf-8')
        
        # Create data URI
        mime_type = "image/png" if format.upper() == "PNG" else "image/jpeg"
        data_uri = f"data:{mime_type};base64,{base64_data}"
        
        self.log_debug(f"Created data URI: {len(data_uri)} chars")
        return data_uri
    
    def cleanup_temp_files(self, *file_paths):
        """Clean up temporary files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                    self.log_debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    self.log_error(f"Could not delete temp file {file_path}: {e}")
    
    def file_to_base64(self, file_path: str) -> str:
        """Convert file to base64 string"""
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')
                self.log_debug(f"Converted file to base64: {len(base64_data)} chars")
                return base64_data
        except Exception as e:
            self.log_error(f"Failed to convert file to base64: {e}")
            raise

    def prepare_file_payload(self, file_path: str, use_base64: bool = False) -> Dict[str, Any]:
        """Prepare file payload for API - try different formats"""
        if file_path is None:
            return None
            
        if use_base64:
            # Try base64 format with different structure
            base64_data = self.file_to_base64(file_path)
            return {
                "data": base64_data,
                "meta": {"_type": "gradio.FileData"},
                "name": os.path.basename(file_path),
                "is_file": False
            }
        else:
            # Try local file path format
            return {
                "path": file_path,
                "meta": {"_type": "gradio.FileData"}
            }

    def prepare_file_payload_v2(self, file_path: str) -> Dict[str, Any]:
        """Alternative file payload format for API"""
        if file_path is None:
            return None
            
        # Try a different format that might work
        base64_data = self.file_to_base64(file_path)
        return {
            "data": f"data:image/png;base64,{base64_data}",
            "meta": {"_type": "gradio.FileData"}
        }

    def prepare_file_payload_v3(self, file_path: str) -> Dict[str, Any]:
        """Another alternative file payload format for API"""
        if file_path is None:
            return None
            
        # Try with just the base64 data as a string
        base64_data = self.file_to_base64(file_path)
        return base64_data

    def make_api_request(self, payload: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Make API request with retry logic"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'ComfyUI-MirrorNode/1.0'
            }
            
            # Debug: Log the payload structure
            self.log_debug(f"API Payload structure:")
            
            # Check if this is a base64 payload or path payload
            is_base64 = 'data' in payload['data'][0] and 'path' not in payload['data'][0]
            is_data_uri = isinstance(payload['data'][0], str) and payload['data'][0].startswith('data:')
            is_simple_base64 = isinstance(payload['data'][0], str) and not payload['data'][0].startswith('data:')
            is_path_data_uri = isinstance(payload['data'][0], dict) and 'path' in payload['data'][0] and payload['data'][0]['path'].startswith('data:')
            
            if is_path_data_uri:
                self.log_debug(f"  - Base image: data URI in path")
                self.log_debug(f"  - Product image: data URI in path")
                self.log_debug(f"  - Workflow type: {payload['data'][2]}")
                self.log_debug(f"  - Mask: {payload['data'][3]}")
                self.log_debug(f"  - Quality: {payload['data'][4]}")
                self.log_debug(f"  - API key length: {len(payload['data'][5])}")
            elif is_data_uri:
                self.log_debug(f"  - Base image: data URI format")
                self.log_debug(f"  - Product image: data URI format")
                self.log_debug(f"  - Workflow type: {payload['data'][2]}")
                self.log_debug(f"  - Mask: {payload['data'][3]}")
                self.log_debug(f"  - Quality: {payload['data'][4]}")
                self.log_debug(f"  - API key length: {len(payload['data'][5])}")
            elif is_simple_base64:
                self.log_debug(f"  - Base image: simple base64 string")
                self.log_debug(f"  - Product image: simple base64 string")
                self.log_debug(f"  - Workflow type: {payload['data'][2]}")
                self.log_debug(f"  - Mask: {payload['data'][3]}")
                self.log_debug(f"  - Quality: {payload['data'][4]}")
                self.log_debug(f"  - API key length: {len(payload['data'][5])}")
            elif is_base64:
                self.log_debug(f"  - Base image: base64 data ({len(payload['data'][0]['data'])} chars)")
                self.log_debug(f"  - Product image: base64 data ({len(payload['data'][1]['data'])} chars)")
                self.log_debug(f"  - Workflow type: {payload['data'][2]}")
                self.log_debug(f"  - Mask: {payload['data'][3]}")
                self.log_debug(f"  - Quality: {payload['data'][4]}")
                self.log_debug(f"  - API key length: {len(payload['data'][5])}")
            else:
                self.log_debug(f"  - Base image path: {payload['data'][0]['path']}")
                self.log_debug(f"  - Product image path: {payload['data'][1]['path']}")
                self.log_debug(f"  - Workflow type: {payload['data'][2]}")
                self.log_debug(f"  - Mask path: {payload['data'][3]}")
                self.log_debug(f"  - Quality: {payload['data'][4]}")
                self.log_debug(f"  - API key length: {len(payload['data'][5])}")
            
            # Debug: Check if files exist before sending (only for path payloads)
            if not is_base64 and not is_data_uri and not is_simple_base64 and not is_path_data_uri:
                base_path = payload['data'][0]['path']
                product_path = payload['data'][1]['path']
                mask_path = payload['data'][3]['path'] if payload['data'][3] else None
                
                self.log_debug(f"File existence check:")
                self.log_debug(f"  - Base file exists: {os.path.exists(base_path)}")
                self.log_debug(f"  - Product file exists: {os.path.exists(product_path)}")
                if mask_path:
                    self.log_debug(f"  - Mask file exists: {os.path.exists(mask_path)}")
            
            # Debug: Log full payload structure for troubleshooting
            self.log_debug(f"Full payload structure:")
            for i, item in enumerate(payload['data']):
                if item is None:
                    self.log_debug(f"  [{i}]: None")
                elif isinstance(item, dict):
                    if is_base64:
                        if 'data' in item:
                            self.log_debug(f"  [{i}]: base64 data ({len(item['data'])} chars)")
                        else:
                            self.log_debug(f"  [{i}]: {list(item.keys())}")
                    elif is_path_data_uri:
                        if 'path' in item and item['path'].startswith('data:'):
                            self.log_debug(f"  [{i}]: data URI in path ({len(item['path'])} chars)")
                        else:
                            self.log_debug(f"  [{i}]: {list(item.keys())}")
                    else:
                        self.log_debug(f"  [{i}]: {list(item.keys())}")
                elif isinstance(item, str):
                    if item.startswith('data:'):
                        self.log_debug(f"  [{i}]: data URI ({len(item)} chars)")
                    else:
                        self.log_debug(f"  [{i}]: base64 string ({len(item)} chars)")
                else:
                    self.log_debug(f"  [{i}]: {type(item).__name__} = {item}")
            
            self.log_info(f"Making API request to: {self.api_base_url}/generate")
            response = requests.post(
                f"{self.api_base_url}/generate",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            self.log_info(f"API Response Status: {response.status_code}")
            
            # Handle different response scenarios
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    raise Exception(f"API Error: {result['error']}")
                self.log_info("API request successful!")
                return result
            elif response.status_code == 429:  # Rate limit
                if retry_count < self.max_retries:
                    wait_time = (2 ** retry_count) * 5  # Exponential backoff
                    self.log_info(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    return self.make_api_request(payload, retry_count + 1)
                else:
                    raise Exception("Rate limit exceeded. Please wait before making more requests.")
            elif response.status_code == 400 and "unsupported file path format" in response.text:
                # Try with base64 format if we get this specific error
                self.log_info("Got unsupported file path format error, trying with base64...")
                if retry_count < 3:  # Allow up to 3 retries with different formats
                    return self.make_api_request_with_base64(payload, retry_count + 1)
                else:
                    raise Exception("API doesn't support local file paths or any base64 format")
            elif response.status_code == 400 and "invalid image data format" in response.text:
                # Try with different base64 format if we get this error
                self.log_info("Got invalid image data format error, trying different base64 format...")
                if retry_count < 3:  # Allow up to 3 retries with different formats
                    return self.make_api_request_with_base64(payload, retry_count + 1)
                else:
                    raise Exception("API doesn't accept any of the tried image formats")
            else:
                # Debug: Log the full error response
                self.log_error(f"API Response Status: {response.status_code}")
                self.log_error(f"API Response Headers: {dict(response.headers)}")
                self.log_error(f"API Response Text: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                self.log_info(f"Request timeout. Retrying... ({retry_count + 1}/{self.max_retries})")
                return self.make_api_request(payload, retry_count + 1)
            else:
                raise Exception("Request timed out after multiple retries")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")

    def make_api_request_with_base64(self, original_payload: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Make API request using base64 format for files"""
        try:
            # Debug: Log the original payload structure
            self.log_info("Converting payload to base64 format...")
            self.log_debug(f"Original payload data length: {len(original_payload['data'])}")
            
            # Extract file paths from original payload
            base_path = original_payload['data'][0]['path']
            product_path = original_payload['data'][1]['path']
            
            # Handle mask path carefully - it might be None
            mask_path = None
            if len(original_payload['data']) > 3 and original_payload['data'][3] is not None:
                mask_path = original_payload['data'][3]['path']
            
            self.log_debug(f"Extracted paths:")
            self.log_debug(f"  - Base path: {base_path}")
            self.log_debug(f"  - Product path: {product_path}")
            self.log_debug(f"  - Mask path: {mask_path}")
            
            # Try different payload formats based on retry count
            if retry_count == 1:
                # Try format v2 (with data URI)
                self.log_info("Trying base64 format v2 (with data URI)...")
                new_payload = {
                    "data": [
                        self.prepare_file_payload_v2(base_path),
                        self.prepare_file_payload_v2(product_path),
                        original_payload['data'][2],  # workflow_type
                        self.prepare_file_payload_v2(mask_path) if mask_path else None,
                        original_payload['data'][4],  # quality
                        original_payload['data'][5]   # api_key
                    ]
                }
            elif retry_count == 2:
                # Try format v3 (just base64 string)
                self.log_info("Trying base64 format v3 (just base64 string)...")
                new_payload = {
                    "data": [
                        self.prepare_file_payload_v3(base_path),
                        self.prepare_file_payload_v3(product_path),
                        original_payload['data'][2],  # workflow_type
                        self.prepare_file_payload_v3(mask_path) if mask_path else None,
                        original_payload['data'][4],  # quality
                        original_payload['data'][5]   # api_key
                    ]
                }
            else:
                # Try original base64 format
                self.log_info("Trying base64 format v1...")
                new_payload = {
                    "data": [
                        self.prepare_file_payload(base_path, use_base64=True),
                        self.prepare_file_payload(product_path, use_base64=True),
                        original_payload['data'][2],  # workflow_type
                        self.prepare_file_payload(mask_path, use_base64=True) if mask_path else None,
                        original_payload['data'][4],  # quality
                        original_payload['data'][5]   # api_key
                    ]
                }
            
            self.log_info(f"Retrying with base64 format v{retry_count + 1}...")
            return self.make_api_request(new_payload, retry_count)
            
        except Exception as e:
            self.log_error(f"Error in make_api_request_with_base64: {str(e)}")
            self.log_error(f"Original payload structure: {original_payload}")
            raise Exception(f"Failed to convert to base64 format: {str(e)}")
    
    def download_result_image(self, image_url: str) -> Image.Image:
        """Download the result image from the provided URL"""
        try:
            self.log_info(f"Downloading result image from: {image_url}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            self.log_info(f"Downloaded image: {image.size} pixels")
            return image
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
        
        # Clear previous log messages
        self.log_messages = []
        
        try:
            # Validate inputs
            self.validate_inputs(base_image, product_image, workflow_type, quality, api_key, mask_image)
            
            self.log_info(f"Starting YourMirror.io virtual try-on")
            self.log_debug(f"  - Workflow: {workflow_type}")
            self.log_debug(f"  - Quality: {quality}")
            self.log_debug(f"  - Has mask: {mask_image is not None}")
            
            # Convert tensors to PIL images
            base_pil = self.tensor_to_pil(base_image)
            product_pil = self.tensor_to_pil(product_image)
            mask_pil = self.tensor_to_pil(mask_image) if mask_image is not None else None
            
            self.log_debug(f"Image conversion completed")
            self.log_debug(f"  - Base image size: {base_pil.size}")
            self.log_debug(f"  - Product image size: {product_pil.size}")
            if mask_pil:
                self.log_debug(f"  - Mask image size: {mask_pil.size}")
            
            # Create data URIs directly (this is what the API expects)
            self.log_info("Creating data URIs for API...")
            base_data_uri = self.create_data_uri(base_pil, "PNG")
            product_data_uri = self.create_data_uri(product_pil, "PNG")
            mask_data_uri = self.create_data_uri(mask_pil, "PNG") if mask_pil else None
            
            # Prepare API payload using data URIs
            payload = {
                "data": [
                    {"path": base_data_uri, "meta": {"_type": "gradio.FileData"}},
                    {"path": product_data_uri, "meta": {"_type": "gradio.FileData"}},
                    workflow_type,
                    {"path": mask_data_uri, "meta": {"_type": "gradio.FileData"}} if mask_data_uri else None,
                    quality,
                    api_key.strip()
                ]
            }
            
            # Make API request
            self.log_debug("Sending request to YourMirror.io API...")
            result = self.make_api_request(payload)
            
            # Extract result image URL
            if 'data' not in result or not result['data']:
                raise Exception("No image data returned from API")
            
            image_url = result['data'][0]
            self.log_debug(f"Downloading result from: {image_url}")
            
            # Download and convert result image
            result_pil = self.download_result_image(image_url)
            result_tensor = self.pil_to_tensor(result_pil)
            
            self.log_info("Virtual try-on completed successfully!")
            
            # Create debug logs string
            debug_logs = "\n".join(self.log_messages)
            
            return (result_tensor, debug_logs)
            
        except Exception as e:
            error_msg = str(e)
            self.log_error(f"YourMirror.io Error: {error_msg}")
            
            # Create error image with text
            error_image = Image.new('RGB', (512, 512), color=(220, 53, 69))  # Bootstrap danger red
            result_tensor = self.pil_to_tensor(error_image)
            
            # Create debug logs string
            debug_logs = "\n".join(self.log_messages)
            
            # Re-raise the exception so ComfyUI shows the error
            raise Exception(f"MirrorNode Error: {error_msg}")
            
        finally:
            # Clean up temporary files
            self.cleanup_temp_files(*temp_files)