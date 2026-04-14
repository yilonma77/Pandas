# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['/Users/grtk/Documents/GitHub/Pandas/app.py'],
    pathex=['/Users/grtk/Documents/GitHub/Pandas'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,       # ← pas de fenêtre Terminal
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name='Bienvenue.app',
    bundle_identifier='com.local.bienvenue',
    info_plist={
        'NSHighResolutionCapable': True,
    },
)
