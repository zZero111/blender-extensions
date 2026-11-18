# Blender Extensions

A home for all my Blender extensions. Each extension lives in its own folder under `extensions/`.

## Layout

```
extensions/
  my_extension/
    blender_manifest.toml
    __init__.py
```

## Building an extension

```
blender --command extension build --source-dir extensions/my_extension --output-dir dist
```

Validate with `blender --command extension validate extensions/my_extension`.

## Auto-updating install

Every push to `main` builds all extensions and publishes them to GitHub Pages. In Blender:
Preferences > Get Extensions > Repositories > Add Remote Repository, and use

```
https://zzero111.github.io/blender-extensions/index.json
```

Bump `version` in an extension's `blender_manifest.toml` before pushing so Blender offers the update.
