# ReadTheDocs Setup Guide for SemiDGFEM

**Author: Dr. Mazharuddin Mohammed**  
**Email: mazharuddin.mohammed.official@gmail.com**

## Why Documentation Isn't Showing

The documentation at `semidgfem.readthedocs.io` isn't showing because:

1. **ReadTheDocs Project Not Created**: The project doesn't exist on ReadTheDocs yet
2. **Repository Not Connected**: ReadTheDocs needs to be connected to your GitHub repository
3. **Documentation Not Pushed**: The documentation files need to be in a public repository

## Step-by-Step Setup Instructions

### Step 1: Ensure Repository is Public and Accessible

First, make sure your SemiDGFEM repository is:
- ✅ Public on GitHub/GitLab
- ✅ Contains the documentation files we created
- ✅ Has the latest commits pushed

**Push the documentation commits:**
```bash
cd /home/madmax/Documents/dev/projects/SemiDGFEM
git push origin master
```

### Step 2: Create ReadTheDocs Account

1. **Go to ReadTheDocs**: https://readthedocs.org/
2. **Sign Up/Login** using your GitHub account
3. **Authorize ReadTheDocs** to access your repositories

### Step 3: Import Your Project

1. **Click "Import a Project"** on ReadTheDocs dashboard
2. **Select your SemiDGFEM repository** from the list
3. **Configure project settings**:
   - **Name**: `semidgfem`
   - **Repository URL**: Your GitHub repository URL
   - **Default Branch**: `master` (or `main`)
   - **Language**: `English`
   - **Programming Language**: `Python`

### Step 4: Configure Advanced Settings

In your ReadTheDocs project settings:

1. **Go to Admin → Advanced Settings**
2. **Set the following**:
   - ✅ **Install your project inside a virtualenv using setup.py install**: Checked
   - ✅ **Use system packages**: Unchecked
   - ✅ **Python configuration file**: `docs/conf.py`
   - ✅ **Requirements file**: `docs/requirements.txt`

### Step 5: Trigger First Build

1. **Go to Builds** in your ReadTheDocs project
2. **Click "Build Version"** to trigger the first build
3. **Monitor the build log** for any errors

## Expected File Structure

ReadTheDocs will look for this structure in your repository:

```
SemiDGFEM/
├── .readthedocs.yaml          # ✅ ReadTheDocs configuration
├── docs/
│   ├── conf.py               # ✅ Sphinx configuration
│   ├── index.rst             # ✅ Main documentation
│   ├── tutorials.rst         # ✅ Tutorials
│   ├── python_api.rst        # ✅ API reference
│   ├── requirements.txt      # ✅ Documentation dependencies
│   └── boundary_condition_fixes.md  # ✅ Technical docs
├── CONTRIBUTING.md           # ✅ Contribution guidelines
└── README.md                 # ✅ Project overview
```

## Troubleshooting Common Issues

### Issue 1: Build Fails with Import Errors

**Problem**: ReadTheDocs can't import your Python modules

**Solution**: Update `docs/conf.py` to mock imports:

```python
# Add to docs/conf.py
import sys
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['simulator', 'numpy', 'matplotlib', 'scipy']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
```

### Issue 2: Requirements Installation Fails

**Problem**: Dependencies can't be installed

**Solution**: Simplify `docs/requirements.txt`:

```txt
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=0.18.0
```

### Issue 3: Configuration File Not Found

**Problem**: ReadTheDocs can't find configuration

**Solution**: Ensure `.readthedocs.yaml` is in repository root and properly formatted.

## Manual Setup Alternative

If automatic setup doesn't work, you can manually configure:

### 1. Create ReadTheDocs Project Manually

1. **Go to**: https://readthedocs.org/dashboard/
2. **Click**: "Import Manually"
3. **Fill in**:
   - **Name**: `semidgfem`
   - **Repository URL**: `https://github.com/your-username/SemiDGFEM.git`
   - **Repository type**: `Git`
   - **Default branch**: `master`

### 2. Configure Build Settings

In **Admin → Advanced Settings**:

```yaml
# Configuration
Python configuration file: docs/conf.py
Requirements file: docs/requirements.txt
Python interpreter: CPython 3.9
Install your project inside a virtualenv: Yes
Use system packages: No
```

## Verification Steps

After setup, verify your documentation:

1. **Check Build Status**: https://readthedocs.org/projects/semidgfem/builds/
2. **View Documentation**: https://semidgfem.readthedocs.io/
3. **Test All Pages**:
   - Main index page
   - Tutorials section
   - Python API reference
   - Contribution guidelines

## Expected URLs

Once properly set up, your documentation will be available at:

- **Main Documentation**: https://semidgfem.readthedocs.io/
- **Latest Version**: https://semidgfem.readthedocs.io/en/latest/
- **Stable Version**: https://semidgfem.readthedocs.io/en/stable/
- **PDF Download**: https://semidgfem.readthedocs.io/_/downloads/en/latest/pdf/

## Next Steps After Setup

1. **Add ReadTheDocs Badge** to README.md:
   ```markdown
   [![Documentation Status](https://readthedocs.org/projects/semidgfem/badge/?version=latest)](https://semidgfem.readthedocs.io/en/latest/?badge=latest)
   ```

2. **Configure Webhooks** for automatic builds on push

3. **Set up Multiple Versions** for releases

4. **Enable PDF/ePub Downloads**

## Support

If you encounter issues:

1. **Check ReadTheDocs Build Logs**: Look for specific error messages
2. **ReadTheDocs Documentation**: https://docs.readthedocs.io/
3. **Community Support**: https://github.com/readthedocs/readthedocs.org/issues

## Summary

The documentation files are ready and properly configured. You just need to:

1. ✅ **Push commits to GitHub** (if not already done)
2. ✅ **Create ReadTheDocs account**
3. ✅ **Import your repository**
4. ✅ **Configure project settings**
5. ✅ **Trigger first build**

Once these steps are completed, your documentation will be live at `https://semidgfem.readthedocs.io/`!

---

**Contact**: mazharuddin.mohammed.official@gmail.com  
**Project**: SemiDGFEM - High Performance TCAD Software
