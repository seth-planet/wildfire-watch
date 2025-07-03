#!/usr/bin/env python3
"""Copy WSDL files from onvif package to app directory"""

import os
import shutil
import site

# WSDL files are installed directly in site-packages/wsdl
wsdl_dst = '/app/wsdl'

# Try to find WSDL directory in site-packages
for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
    if site_dir:
        wsdl_src = os.path.join(site_dir, 'wsdl')
        if os.path.exists(wsdl_src):
            try:
                shutil.copytree(wsdl_src, wsdl_dst)
                print(f"Copied WSDL files from {wsdl_src} to {wsdl_dst}")
                break
            except Exception as e:
                print(f"ERROR copying WSDL files: {e}")
else:
    print("WARNING: WSDL directory not found in any site-packages location")