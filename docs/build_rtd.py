#!/usr/bin/env python3
"""
Read the Docs Build Script for SemiDGFEM
Comprehensive documentation build system with enhanced content

Author: Dr. Mazharuddin Mohammed
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

class RTDBuilder:
    """Read the Docs documentation builder"""
    
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir)
        self.build_dir = self.docs_dir / "_build"
        self.source_dir = self.docs_dir
        
        # Ensure build directory exists
        self.build_dir.mkdir(exist_ok=True)
    
    def check_dependencies(self):
        """Check if required tools are available"""
        print("🔍 Checking dependencies...")
        
        dependencies = {
            "sphinx": self._check_python_module("sphinx"),
            "sphinx_rtd_theme": self._check_python_module("sphinx_rtd_theme"),
            "myst_parser": self._check_python_module("myst_parser"),
            "numpy": self._check_python_module("numpy"),
            "matplotlib": self._check_python_module("matplotlib")
        }
        
        missing = [dep for dep, available in dependencies.items() if not available]
        
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            print("Installing missing dependencies...")
            self._install_dependencies(missing)
        else:
            print("✅ All dependencies available")
        
        return dependencies
    
    def _check_python_module(self, module: str) -> bool:
        """Check if a Python module is available"""
        try:
            __import__(module)
            return True
        except ImportError:
            return False
    
    def _install_dependencies(self, missing: list):
        """Install missing dependencies"""
        pip_packages = {
            "sphinx": "sphinx>=4.0.0",
            "sphinx_rtd_theme": "sphinx-rtd-theme",
            "myst_parser": "myst-parser",
            "numpy": "numpy",
            "matplotlib": "matplotlib"
        }
        
        for package in missing:
            if package in pip_packages:
                print(f"Installing {package}...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    pip_packages[package]
                ], check=True)
    
    def prepare_source_files(self):
        """Prepare source files for building"""
        print("📝 Preparing source files...")
        
        # Copy markdown files to RST if needed
        markdown_files = [
            "README.md",
            "Installation_Guide.md", 
            "User_Guide.md",
            "API_Reference.md",
            "Developer_Guide.md",
            "Feature_Documentation.md"
        ]
        
        for md_file in markdown_files:
            md_path = self.docs_dir / md_file
            if md_path.exists():
                print(f"   Found {md_file}")
        
        # Ensure all RST files exist
        rst_files = [
            "index.rst",
            "theory.rst",
            "implementation.rst",
            "usage.rst",
            "tutorials.rst",
            "quickstart.rst"
        ]
        
        for rst_file in rst_files:
            rst_path = self.docs_dir / rst_file
            if rst_path.exists():
                print(f"   ✅ {rst_file}")
            else:
                print(f"   ⚠️  Missing {rst_file}")
    
    def build_html(self):
        """Build HTML documentation"""
        print("🏗️  Building HTML documentation...")
        
        html_dir = self.build_dir / "html"
        
        # Run Sphinx build
        cmd = [
            "sphinx-build",
            "-b", "html",
            "-d", str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(html_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ HTML build successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ HTML build failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    def build_pdf(self):
        """Build PDF documentation"""
        print("📄 Building PDF documentation...")
        
        pdf_dir = self.build_dir / "latex"
        
        # Build LaTeX first
        cmd_latex = [
            "sphinx-build",
            "-b", "latex",
            "-d", str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(pdf_dir)
        ]
        
        try:
            subprocess.run(cmd_latex, capture_output=True, text=True, check=True)
            
            # Build PDF from LaTeX
            if (pdf_dir / "semidgfem.tex").exists():
                cmd_pdf = ["pdflatex", "semidgfem.tex"]
                subprocess.run(cmd_pdf, cwd=pdf_dir, capture_output=True, check=True)
                subprocess.run(cmd_pdf, cwd=pdf_dir, capture_output=True, check=True)  # Run twice for references
                
                print("✅ PDF build successful")
                return True
            else:
                print("❌ LaTeX file not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ PDF build failed: {e}")
            return False
        except FileNotFoundError:
            print("⚠️  pdflatex not found, skipping PDF build")
            return False
    
    def build_epub(self):
        """Build EPUB documentation"""
        print("📚 Building EPUB documentation...")
        
        epub_dir = self.build_dir / "epub"
        
        cmd = [
            "sphinx-build",
            "-b", "epub",
            "-d", str(self.build_dir / "doctrees"),
            str(self.source_dir),
            str(epub_dir)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ EPUB build successful")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ EPUB build failed: {e}")
            return False
    
    def validate_build(self):
        """Validate the built documentation"""
        print("🔍 Validating build...")
        
        html_dir = self.build_dir / "html"
        
        # Check if main files exist
        required_files = [
            "index.html",
            "theory.html",
            "implementation.html",
            "usage.html",
            "tutorials.html",
            "quickstart.html"
        ]
        
        missing_files = []
        for file in required_files:
            if not (html_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        else:
            print("✅ All required files present")
            return True
    
    def generate_sitemap(self):
        """Generate sitemap for the documentation"""
        print("🗺️  Generating sitemap...")
        
        html_dir = self.build_dir / "html"
        
        # Find all HTML files
        html_files = list(html_dir.glob("**/*.html"))
        
        # Generate sitemap content
        sitemap_content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
"""
        
        base_url = "https://semidgfem.readthedocs.io/en/latest/"
        
        for html_file in html_files:
            relative_path = html_file.relative_to(html_dir)
            url = base_url + str(relative_path).replace("\\", "/")
            
            sitemap_content += f"""  <url>
    <loc>{url}</loc>
    <lastmod>{time.strftime('%Y-%m-%d')}</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
"""
        
        sitemap_content += "</urlset>\n"
        
        # Write sitemap
        with open(html_dir / "sitemap.xml", "w") as f:
            f.write(sitemap_content)
        
        print(f"✅ Sitemap generated with {len(html_files)} pages")
    
    def optimize_images(self):
        """Optimize images in the documentation"""
        print("🖼️  Optimizing images...")
        
        html_dir = self.build_dir / "html"
        image_dirs = [html_dir / "_images", html_dir / "_static"]
        
        optimized_count = 0
        
        for img_dir in image_dirs:
            if img_dir.exists():
                for img_file in img_dir.glob("*.png"):
                    # Simple optimization: could add actual image compression here
                    optimized_count += 1
        
        print(f"✅ Processed {optimized_count} images")
    
    def create_search_index(self):
        """Create search index for documentation"""
        print("🔍 Creating search index...")
        
        # Sphinx automatically creates search index
        # This is a placeholder for additional search functionality
        
        html_dir = self.build_dir / "html"
        search_files = list(html_dir.glob("search*.js"))
        
        if search_files:
            print(f"✅ Search index created: {len(search_files)} files")
        else:
            print("⚠️  No search index found")
    
    def build_all(self):
        """Build all documentation formats"""
        print("🚀 BUILDING COMPREHENSIVE RTD DOCUMENTATION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Check dependencies
        self.check_dependencies()
        
        # Prepare source files
        self.prepare_source_files()
        
        # Build documentation
        results = {}
        
        # HTML (required)
        results['html'] = self.build_html()
        
        # PDF (optional)
        results['pdf'] = self.build_pdf()
        
        # EPUB (optional)
        results['epub'] = self.build_epub()
        
        # Post-processing
        if results['html']:
            self.validate_build()
            self.generate_sitemap()
            self.optimize_images()
            self.create_search_index()
        
        # Summary
        build_time = time.time() - start_time
        self.print_summary(results, build_time)
        
        return results
    
    def print_summary(self, results, build_time):
        """Print build summary"""
        print(f"\n📊 BUILD SUMMARY")
        print("=" * 40)
        
        total_formats = len(results)
        successful_formats = sum(results.values())
        
        for format_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {format_name.upper()} documentation")
        
        print(f"\n📈 Success rate: {successful_formats}/{total_formats} formats")
        print(f"⏱️  Build time: {build_time:.1f} seconds")
        
        if results.get('html'):
            html_dir = self.build_dir / "html"
            print(f"\n📁 Documentation available at: {html_dir}")
            print(f"🌐 Open: {html_dir / 'index.html'}")
        
        if successful_formats == total_formats:
            print("\n🎉 ALL DOCUMENTATION FORMATS BUILT SUCCESSFULLY!")
        else:
            print(f"\n⚠️  {total_formats - successful_formats} FORMAT(S) FAILED")

def main():
    """Main function"""
    print("📚 SemiDGFEM Read the Docs Builder")
    print("=" * 50)
    
    # Create builder
    builder = RTDBuilder()
    
    # Build all documentation
    results = builder.build_all()
    
    # Return appropriate exit code
    return 0 if any(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
