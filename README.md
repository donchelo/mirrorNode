MirrorNode - ComfyUI YourMirror Virtual Try-On
A custom node for ComfyUI that integrates with YourMirror.io API to provide AI-powered virtual try-on functionality for garments and accessories.

Mostrar imagen
Mostrar imagen
Mostrar imagen

🌟 Features
Multiple Garment Types: Support for eyewear, footwear, dresses, tops, and bottoms
Quality Options: Choose between normal (fast) and high quality (2x cost)
Optional Mask Support: Use masks for better control over try-on areas
Robust Error Handling: Automatic retry logic and comprehensive error messages
Clean Resource Management: Automatic cleanup of temporary files
Rate Limit Handling: Smart retry with exponential backoff
🚀 Installation
Method 1: Via ComfyUI Manager (Recommended)
Install ComfyUI Manager
Search for "MirrorNode" or "YourMirror" in the manager
Click install and restart ComfyUI
Method 2: Manual Installation
Navigate to your ComfyUI custom_nodes directory
Clone this repository:
bash
git clone https://github.com/donchelo/mirrorNode.git
Install dependencies:
bash
cd mirrorNode
pip install -r requirements.txt
Restart ComfyUI
Method 3: ComfyUI Portable
If you're using the portable version of ComfyUI:

bash
cd ComfyUI_windows_portable
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\mirrorNode\requirements.txt
🔑 API Key Setup
Sign up at yourmirror.io
Navigate to the Studio App
Create your API key
⚠️ Important: Save your API key securely - it's only shown once!
Enter your API key in the node's API key field
📖 Usage
Add the Node: Find "YourMirror Virtual Try-On" in the YourMirror category
Connect Inputs:
Base Image: Photo of the person
Product Image: Image of the garment/accessory to try on
Mask Image (optional): Black/white mask for precise control
Configure Settings:
Workflow Type: Select the appropriate garment type
Quality: Choose normal or high quality
API Key: Enter your YourMirror.io API key
Execute: Run your workflow and get the virtual try-on result
👕 Supported Workflow Types
Type	Description	Use Case
eyewear	Sunglasses, glasses	Face accessories
footwear	Shoes, boots, sneakers	Footwear try-on
dress	Full dresses	Full-body garments
top	Shirts, blouses, jackets	Upper body garments
bottom	Pants, skirts, shorts	Lower body garments
⚙️ Quality Settings
Normal (16 steps): Faster processing, standard quality, 1 API credit
High (40 steps): Better quality, slower processing, 2 API credits
📊 API Limits & Pricing
Rate Limit: 1,000 requests per API key per hour
Normal Quality: 1 credit per request
High Quality: 2 credits per request
🖼️ Supported Image Formats
Fully Supported: JPG, JPEG, PNG, WEBP
Auto-Converted: AVIF, BMP, GIF, TIFF, ICO (automatically converted to JPG)
🛠️ Troubleshooting
Common Issues
"API Key is required"
Ensure you've entered your API key from yourmirror.io
Check that the key doesn't have extra spaces
"Rate limit exceeded"
Wait for the hourly limit to reset
Consider using normal quality to preserve credits
"NSFW content detected"
Use different product images that comply with content policies
Avoid images with inappropriate content
"Network error" or timeouts
Check your internet connection
The node automatically retries failed requests
Ensure yourmirror.io is accessible from your network
Debug Information
Check the ComfyUI console for detailed error messages and processing status. The node provides comprehensive logging for troubleshooting.

🔧 Configuration
Optional configuration file config.json.example is provided for advanced users:

json
{
  "api_base_url": "https://apiservice.yourmirror.io",
  "timeout": 60,
  "max_retries": 3,
  "default_quality": "normal"
}
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Credits
YourMirror.io - For providing the excellent virtual try-on API
ComfyUI Community - For the amazing framework and ecosystem
Contributors - Thank you to all contributors and testers
📞 Support
Issues: GitHub Issues
Discussions: GitHub Discussions
API Support: YourMirror.io Support
📈 Changelog
v1.0.0 (Initial Release)
✅ Support for all YourMirror.io workflow types
✅ Quality settings (normal/high)
✅ Optional mask image support
✅ Comprehensive error handling and retry logic
✅ Automatic resource cleanup
✅ Rate limiting with exponential backoff
Made with ❤️ for the ComfyUI community

