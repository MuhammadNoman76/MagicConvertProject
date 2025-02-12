import streamlit as st
import requests
from typing import Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import tempfile
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data Classes and Abstract Classes
@dataclass
class BillResponse:
    """Data class to handle bill processing responses."""
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None

class TextExtractor(ABC):
    """Abstract base class for text extraction implementations."""
    @abstractmethod
    def extract_text(self, image_path: str) -> Optional[str]:
        """Extract text from an image file."""
        pass

class MagicConvertExtractor(TextExtractor):
    """Implementation of TextExtractor using MagicConvert."""
    def __init__(self):
        try:
            from MagicConvert import MagicConvert
            self.converter = MagicConvert()
        except ImportError:
            logger.error("MagicConvert package not found. Please install it first.")
            raise

    def extract_text(self, image_path: str) -> Optional[str]:
        """Extract text from image using MagicConvert."""
        try:
            result = self.converter.magic(image_path)
            return result.get_text if result else None
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None

class BillExtractor:
    """Main class for processing bills and extracting structured data."""
    
    def __init__(
        self, 
        openrouter_token: str,
        text_extractor: Optional[TextExtractor] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    ):
        self.openrouter_token = openrouter_token
        self.base_url = base_url
        self.text_extractor = text_extractor or MagicConvertExtractor()

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers for the API request."""
        return {
            "Authorization": f"Bearer {self.openrouter_token}",
            "HTTP-Referer": "localhost:3000",
            "X-Title": "BillReader",
            "Content-Type": "application/json"
        }

    def _create_prompt(self, content: str) -> str:
        """Create the prompt for the language model."""
        return f"""You are a bill formatting specialist. Convert the following bill content into a clean, structured Markdown format following these exact specifications:

1. Header Section:
   - Company name and details
   - Company contact information
   - Company address

2. Client Information:
   - Client name
   - Invoice number
   - Client address
   - Client contact details
   - Invoice date and due date

3. Item Details:
   - Create a Markdown table with columns: Item Description | Price | Quantity | Total
   - Align numbers to the right in the table
   - Use proper table formatting with | separators

4. Financial Summary:
   - Subtotal
   - Tax amount (if any)
   - Discounts (if any)
   - Final total amount

5. Payment Information:
   - Bank details
   - Account holder name
   - Account/reference numbers

6. Additional Details:
   - Terms and conditions
   - Any additional notes or references

Format Rules:
- Use ## for main sections and ### for subsections
- Use **bold** for important fields and values
- Add horizontal lines (---) between major sections
- Maintain consistent spacing and alignment
- Preserve all numerical values exactly as shown
- Format currency with appropriate symbols

Input Bill:
{content}

Provide only the formatted Markdown output without any explanations or additional text."""

    def _call_llm(self, text: str) -> BillResponse:
        """Make API call to the language model."""
        try:
            payload = {
                "model": "meta-llama/llama-3-8b-instruct:free",
                "messages": [
                    {"role": "user", "content": self._create_prompt(text)}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }

            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return BillResponse(
                    success=True,
                    data=data['choices'][0]['message']['content']
                )
            
            return BillResponse(
                success=False,
                error="No valid response from LLM"
            )

        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return BillResponse(success=False, error="Request timed out")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.text}")
            return BillResponse(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return BillResponse(success=False, error=str(e))

    def process_bill_image(self, image_path: str) -> BillResponse:
        """Process a bill image and return structured data."""
        try:
            if not Path(image_path).is_file():
                return BillResponse(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )
            text_result = self.text_extractor.extract_text(image_path)
            if not text_result:
                return BillResponse(
                    success=False,
                    error="Failed to extract text from image"
                )
            return self._call_llm(text_result)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return BillResponse(success=False, error=str(e))

def main():
    st.set_page_config(
        page_title="Smart Bill Processor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Fixed CSS for proper dark/light mode theming
    st.markdown("""
        <style>
        /* Base styles */
        body {
            --primary-color: #0d6efd;
            --background-color: #ffffff;
            --text-color: #000000;
            --secondary-background: #f8f9fa;
            --border-color: #e0e0e0;
        }

        /* Dark mode overrides */
        @media (prefers-color-scheme: dark) {
            body {
                --background-color: #0e1117;
                --text-color: #ffffff;
                --secondary-background: #1e1e1e;
                --border-color: #464646;
            }
        }

        [data-testid="stAppViewContainer"] {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        [data-testid="stSidebar"] {
            background-color: var(--secondary-background) !important;
        }

        /* Text elements */
        h1, h2, h3, h4, h5, h6, p, div, span, pre, label {
            color: var(--text-color) !important;
        }

        /* Containers */
        .upload-container, .result-container {
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background-color: var(--secondary-background);
            border: 1px solid var(--border-color);
        }

        /* File uploader */
        .stFileUploader>div {
            background-color: var(--secondary-background) !important;
            border: 2px dashed var(--border-color) !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--primary-color) !important;
            color: white !important;
            border: none !important;
        }

        /* Tables */
        table {
            background-color: var(--secondary-background) !important;
            color: var(--text-color) !important;
        }

        /* Code blocks */
        pre code {
            background-color: var(--secondary-background) !important;
            border: 1px solid var(--border-color) !important;
        }

        /* Additional spacing */
        .stMarkdown {
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # App Header
    st.markdown("""
        <div class='fadeIn' style='text-align: center;'>
            <h1>📑 Smart Bill Processor</h1>
            <p style='font-size: 1.2rem; margin-bottom: 2rem;'>
                Transform your bills into beautifully structured documents instantly
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2>⚙️ Settings & Upload</h2>
            </div>
        """, unsafe_allow_html=True)
        
        openrouter_token = st.text_input(
            "OpenRouter API Token",
            type="password",
            help="Enter your OpenRouter API token here",
            placeholder="Enter your API token..."
        )
        
        st.markdown("<div style='margin: 1.5rem 0;'><hr></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div class='upload-container'>
                <h3>📤 Upload Bill</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Drop your bill image here",
            type=["png", "jpg", "jpeg", "pdf"],
            help="Supported formats: PNG, JPG, JPEG, PDF"
        )
        
        generate_button = st.button("🚀 Generate Bill Analysis", use_container_width=True)
        
        if generate_button:
            if not openrouter_token:
                st.error("⚠️ Please enter your OpenRouter API token")
            elif not uploaded_file:
                st.error("⚠️ Please upload a bill image")
            else:
                with st.spinner("🔄 Processing your bill..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        extractor = BillExtractor(openrouter_token)
                        result = extractor.process_bill_image(tmp_path)
                        os.unlink(tmp_path)

                        if result.success:
                            st.success("✨ Bill processed successfully!")
                            st.session_state.markdown_result = result.data
                        else:
                            st.error(f"❌ Processing failed: {result.error}")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("<div style='margin: 1.5rem 0;'><hr></div>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='margin-top: 1rem;'>
                <h4>📋 Quick Guide</h4>
                <ol style='margin-left: 1rem;'>
                    <li>Enter your API token</li>
                    <li>Upload your bill image</li>
                    <li>Click Generate</li>
                    <li>Download in your preferred format</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    # Main Content Area
    st.markdown("""
        <div class='result-container'>
            <h3>📋 Processed Result</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'markdown_result' in st.session_state:
        with st.container():
            st.markdown(st.session_state.markdown_result)
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col2:
                st.download_button(
                    "📥 Download Markdown",
                    st.session_state.markdown_result,
                    file_name="processed_bill.md",
                    mime="text/markdown",
                    key="markdown_download",
                    help="Download as Markdown file",
                    use_container_width=True
                )
    else:
        st.markdown("""
            <div class='placeholder-content'>
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" fill="currentColor" viewBox="0 0 16 16" style="margin-bottom: 1rem;">
                    <path d="M4 0h5.293A1 1 0 0 1 10 .293L13.707 4a1 1 0 0 1 .293.707V14a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V2a2 2 0 0 1 2-2zm5.5 1.5v2a1 1 0 0 0 1 1h2l-3-3z"/>
                </svg>
                <h4 style='margin-bottom: 0.5rem;'>No Bill Processed Yet</h4>
                <p style='margin-bottom: 0;'>Upload a bill and click Generate to see the results here</p>
            </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div class='footer'>
            <p>
                Made with ❤️ by Muhammad Noman<br>
                <a href="https://github.com/MuhammadNoman76/MagicConvertProject" target="_blank">GitHub</a> | 
                <a href="https://pypi.org/project/MagicConvert/" target="_blank">Documentation</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()