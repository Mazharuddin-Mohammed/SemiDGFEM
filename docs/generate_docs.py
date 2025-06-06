#!/usr/bin/env python3
"""
Documentation Generation Script for SemiDGFEM
Generates comprehensive documentation in multiple formats

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
import time

class DocumentationGenerator:
    """Comprehensive documentation generator"""
    
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir)
        self.output_dir = self.docs_dir / "generated"
        self.formats = ["html", "pdf", "epub"]
        
        # Documentation files
        self.doc_files = [
            "README.md",
            "Installation_Guide.md",
            "User_Guide.md",
            "API_Reference.md",
            "Developer_Guide.md",
            "Feature_Documentation.md"
        ]
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required tools are available"""
        dependencies = {
            "pandoc": self._check_command("pandoc --version"),
            "pdflatex": self._check_command("pdflatex --version"),
            "python": self._check_command("python3 --version"),
            "markdown": self._check_python_module("markdown"),
            "weasyprint": self._check_python_module("weasyprint")
        }
        
        return dependencies
    
    def _check_command(self, command: str) -> bool:
        """Check if a command is available"""
        try:
            subprocess.run(command.split(), capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_python_module(self, module: str) -> bool:
        """Check if a Python module is available"""
        try:
            __import__(module)
            return True
        except ImportError:
            return False
    
    def generate_html_docs(self) -> bool:
        """Generate HTML documentation"""
        print("📄 Generating HTML documentation...")
        
        try:
            html_dir = self.output_dir / "html"
            html_dir.mkdir(exist_ok=True)
            
            # Copy CSS and assets
            self._copy_assets(html_dir)
            
            # Generate individual HTML files
            for doc_file in self.doc_files:
                input_file = self.docs_dir / doc_file
                output_file = html_dir / doc_file.replace('.md', '.html')
                
                if input_file.exists():
                    self._convert_to_html(input_file, output_file)
                    print(f"   ✅ Generated {output_file.name}")
            
            # Generate index page
            self._generate_html_index(html_dir)
            
            print(f"   📁 HTML documentation saved to: {html_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ HTML generation failed: {e}")
            return False
    
    def generate_pdf_docs(self) -> bool:
        """Generate PDF documentation"""
        print("📄 Generating PDF documentation...")
        
        try:
            pdf_dir = self.output_dir / "pdf"
            pdf_dir.mkdir(exist_ok=True)
            
            # Generate individual PDFs
            for doc_file in self.doc_files:
                input_file = self.docs_dir / doc_file
                output_file = pdf_dir / doc_file.replace('.md', '.pdf')
                
                if input_file.exists():
                    self._convert_to_pdf(input_file, output_file)
                    print(f"   ✅ Generated {output_file.name}")
            
            # Generate combined PDF
            self._generate_combined_pdf(pdf_dir)
            
            print(f"   📁 PDF documentation saved to: {pdf_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ PDF generation failed: {e}")
            return False
    
    def generate_epub_docs(self) -> bool:
        """Generate EPUB documentation"""
        print("📄 Generating EPUB documentation...")
        
        try:
            epub_dir = self.output_dir / "epub"
            epub_dir.mkdir(exist_ok=True)
            
            # Combine all markdown files
            combined_md = self._combine_markdown_files()
            
            # Convert to EPUB
            output_file = epub_dir / "SemiDGFEM_Documentation.epub"
            self._convert_to_epub(combined_md, output_file)
            
            print(f"   ✅ Generated {output_file.name}")
            print(f"   📁 EPUB documentation saved to: {epub_dir}")
            return True
            
        except Exception as e:
            print(f"   ❌ EPUB generation failed: {e}")
            return False
    
    def _copy_assets(self, html_dir: Path):
        """Copy CSS and other assets"""
        assets_dir = html_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        
        # Create CSS file
        css_content = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 2em;
        }
        
        h1 { border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }
        
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #3498db;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            color: #666;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
        }
        
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        
        .nav {
            background-color: #2c3e50;
            color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        
        .nav a {
            color: #3498db;
            text-decoration: none;
            margin-right: 20px;
        }
        
        .nav a:hover {
            text-decoration: underline;
        }
        
        .status-complete { color: #27ae60; font-weight: bold; }
        .status-partial { color: #f39c12; font-weight: bold; }
        .status-todo { color: #e74c3c; font-weight: bold; }
        """
        
        with open(assets_dir / "style.css", "w") as f:
            f.write(css_content)
    
    def _convert_to_html(self, input_file: Path, output_file: Path):
        """Convert markdown to HTML"""
        try:
            # Use pandoc if available
            if self._check_command("pandoc --version"):
                cmd = [
                    "pandoc",
                    str(input_file),
                    "-o", str(output_file),
                    "--standalone",
                    "--css", "assets/style.css",
                    "--toc",
                    "--toc-depth", "3",
                    "--highlight-style", "github"
                ]
                subprocess.run(cmd, check=True)
            else:
                # Fallback to Python markdown
                self._markdown_to_html_python(input_file, output_file)
                
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Pandoc failed for {input_file.name}, using fallback")
            self._markdown_to_html_python(input_file, output_file)
    
    def _markdown_to_html_python(self, input_file: Path, output_file: Path):
        """Convert markdown to HTML using Python"""
        try:
            import markdown
            
            with open(input_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            md = markdown.Markdown(extensions=['toc', 'codehilite', 'tables'])
            html_content = md.convert(md_content)
            
            # Wrap in HTML template
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{input_file.stem}</title>
                <link rel="stylesheet" href="assets/style.css">
            </head>
            <body>
                <div class="nav">
                    <a href="README.html">Home</a>
                    <a href="Installation_Guide.html">Installation</a>
                    <a href="User_Guide.html">User Guide</a>
                    <a href="API_Reference.html">API Reference</a>
                    <a href="Developer_Guide.html">Developer Guide</a>
                    <a href="Feature_Documentation.html">Features</a>
                </div>
                {html_content}
            </body>
            </html>
            """
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_html)
                
        except ImportError:
            # Basic fallback
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Very basic markdown to HTML conversion
            html_content = content.replace('\n# ', '\n<h1>').replace('\n## ', '\n<h2>')
            html_content = html_content.replace('\n### ', '\n<h3>').replace('\n#### ', '\n<h4>')
            html_content = html_content.replace('\n', '<br>\n')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"<html><body>{html_content}</body></html>")
    
    def _convert_to_pdf(self, input_file: Path, output_file: Path):
        """Convert markdown to PDF"""
        if self._check_command("pandoc --version"):
            cmd = [
                "pandoc",
                str(input_file),
                "-o", str(output_file),
                "--pdf-engine", "pdflatex",
                "--toc",
                "--toc-depth", "3",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt"
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"   ⚠️  Pandoc not available, skipping PDF for {input_file.name}")
    
    def _convert_to_epub(self, input_file: Path, output_file: Path):
        """Convert markdown to EPUB"""
        if self._check_command("pandoc --version"):
            cmd = [
                "pandoc",
                str(input_file),
                "-o", str(output_file),
                "--toc",
                "--toc-depth", "3",
                "--epub-cover-image", "assets/cover.png"  # If available
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"   ⚠️  Pandoc not available, skipping EPUB")
    
    def _combine_markdown_files(self) -> Path:
        """Combine all markdown files into one"""
        combined_file = self.output_dir / "combined.md"
        
        with open(combined_file, 'w', encoding='utf-8') as outfile:
            outfile.write("% SemiDGFEM Documentation\n")
            outfile.write("% Dr. Mazharuddin Mohammed\n")
            outfile.write(f"% {time.strftime('%Y-%m-%d')}\n\n")
            
            for doc_file in self.doc_files:
                input_file = self.docs_dir / doc_file
                if input_file.exists():
                    outfile.write(f"\n\\newpage\n\n")
                    with open(input_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    outfile.write("\n\n")
        
        return combined_file
    
    def _generate_combined_pdf(self, pdf_dir: Path):
        """Generate combined PDF from all documents"""
        combined_md = self._combine_markdown_files()
        output_file = pdf_dir / "SemiDGFEM_Complete_Documentation.pdf"
        
        if self._check_command("pandoc --version"):
            cmd = [
                "pandoc",
                str(combined_md),
                "-o", str(output_file),
                "--pdf-engine", "pdflatex",
                "--toc",
                "--toc-depth", "3",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt",
                "-V", "documentclass=book"
            ]
            subprocess.run(cmd, check=True)
            print(f"   ✅ Generated combined PDF: {output_file.name}")
    
    def _generate_html_index(self, html_dir: Path):
        """Generate HTML index page"""
        index_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>SemiDGFEM Documentation</title>
            <link rel="stylesheet" href="assets/style.css">
        </head>
        <body>
            <h1>SemiDGFEM Documentation</h1>
            <p>Complete documentation for the Semiconductor Discontinuous Galerkin Finite Element Method framework.</p>
            
            <div class="toc">
                <h2>Documentation Sections</h2>
                <ul>
                    <li><a href="README.html">📚 Overview and Quick Start</a></li>
                    <li><a href="Installation_Guide.html">🚀 Installation Guide</a></li>
                    <li><a href="User_Guide.html">📖 User Guide and Tutorials</a></li>
                    <li><a href="API_Reference.html">📋 API Reference</a></li>
                    <li><a href="Developer_Guide.html">🛠️ Developer Guide</a></li>
                    <li><a href="Feature_Documentation.html">🔬 Feature Documentation</a></li>
                </ul>
            </div>
            
            <h2>Framework Highlights</h2>
            <ul>
                <li><span class="status-complete">✅ Complete</span> - Advanced transport models (drift-diffusion, energy, hydrodynamic)</li>
                <li><span class="status-complete">✅ Complete</span> - MOSFET simulation with I-V characterization</li>
                <li><span class="status-complete">✅ Complete</span> - Heterostructure simulation with quantum effects</li>
                <li><span class="status-complete">✅ Complete</span> - GPU acceleration (CUDA/OpenCL)</li>
                <li><span class="status-complete">✅ Complete</span> - SIMD optimization (AVX2/FMA)</li>
                <li><span class="status-complete">✅ Complete</span> - Professional visualization and analysis</li>
            </ul>
        </body>
        </html>
        """
        
        with open(html_dir / "index.html", "w") as f:
            f.write(index_content)
    
    def generate_all_formats(self) -> Dict[str, bool]:
        """Generate documentation in all formats"""
        print("🚀 GENERATING COMPREHENSIVE DOCUMENTATION")
        print("=" * 60)
        
        # Check dependencies
        deps = self.check_dependencies()
        print("📋 Checking dependencies:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"   {status} {dep}")
        
        print()
        
        # Generate documentation
        results = {}
        
        # HTML (always possible)
        results['html'] = self.generate_html_docs()
        
        # PDF (requires pandoc and pdflatex)
        if deps['pandoc'] and deps['pdflatex']:
            results['pdf'] = self.generate_pdf_docs()
        else:
            print("📄 Skipping PDF generation (missing dependencies)")
            results['pdf'] = False
        
        # EPUB (requires pandoc)
        if deps['pandoc']:
            results['epub'] = self.generate_epub_docs()
        else:
            print("📄 Skipping EPUB generation (missing dependencies)")
            results['epub'] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print generation summary"""
        print("\n📊 DOCUMENTATION GENERATION SUMMARY")
        print("=" * 40)
        
        total_formats = len(results)
        successful_formats = sum(results.values())
        
        for format_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {format_name.upper()} documentation")
        
        print(f"\n📈 Success rate: {successful_formats}/{total_formats} formats")
        
        if successful_formats > 0:
            print(f"\n📁 Generated documentation available in: {self.output_dir}")
            
            if results.get('html'):
                print(f"   🌐 HTML: {self.output_dir}/html/index.html")
            if results.get('pdf'):
                print(f"   📄 PDF: {self.output_dir}/pdf/")
            if results.get('epub'):
                print(f"   📚 EPUB: {self.output_dir}/epub/")
        
        print("\n🎉 Documentation generation complete!")

def main():
    """Main function"""
    print("📚 SemiDGFEM Documentation Generator")
    print("=" * 50)
    
    # Create generator
    generator = DocumentationGenerator()
    
    # Generate all formats
    results = generator.generate_all_formats()
    
    # Print summary
    generator.print_summary(results)
    
    return 0 if any(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
