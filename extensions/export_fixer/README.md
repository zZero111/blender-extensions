# Export Fixer

A free Blender extension (4.2 and newer) that catches transform problems before they break your export to Unity, Unreal, Godot or glTF.

## What it checks

| Check | Severity | Why it matters |
|---|---|---|
| Mirrored (negative) scale | Error | Faces show inside-out in most engines and in Substance or Marmoset |
| Non-uniform scale | Warning | Bevel, inset, physics and colliders distort |
| Unapplied scale | Warning | The model arrives at the wrong size |
| Unapplied rotation | Info, off by default | Some pipelines need rotation baked in |
| Scene unit scale is not 1.0 | Warning | Everything exports at the wrong size |
| Zero scale on an axis | Error | The object is flat and its transform can't be applied |

- If a child object inherits a bad scale from its parent, the extension reports it and tells you to fix the parent.
- Linked (library) objects are reported but not fixed.

## What "Fix" does

- **Bakes scale into the mesh.** It also bakes rotation if you turn on *Fixes Also Apply Rotation*. Delta transforms and shape keys are included.
- **Keeps normals pointing outward on mirrored objects.** This is the case where Blender's own Apply Scale leaves them flipped.
- **Leaves child objects where they were.**
- **Copies shared mesh data first.** If other objects use the same mesh, the fixed object gets its own copy so the others aren't changed.
- **Doesn't use `bpy.ops`,** so the fixes work from any editor. Every fix can be undone with Ctrl+Z.

## How to use it

1. Open the 3D Viewport, press **N** and open the **Export Fixer** tab.
2. Pick your target: Unity, Unreal, Godot, glTF or plain FBX.
3. Click **Run Check**.
   - Click the arrow on a row to select that object.
   - Click the tick on a row to fix it, or use **Fix All**.
4. Click **Check & Export**. It checks again, then exports with the target's settings. You can also use *File > Export > Export Fixer: Check & Export*.

The viewport highlight shows problem objects as coloured boxes: red for mirrored, orange for non-uniform scale and yellow for unapplied scale. Turn it on or off with the **Highlight in Viewport** checkbox in the panel.

### Export settings for each target

| Target | Exporter | Settings |
|---|---|---|
| Unity | FBX | Forward -Z, up Y, Apply Scalings: FBX All, Apply Transform on, smoothing: Face, no leaf bones |
| Unreal | FBX | Apply Scalings: FBX Units Scale, smoothing: Face, no leaf bones |
| Godot / glTF | glTF (.glb) | +Y up, modifiers applied |

If a future Blender version renames or removes one of these exporter settings, the extension skips that setting and tells you which one it skipped. It doesn't fail.

## Known limit

Blender has no "before export" hook, so this extension can't run its checks when you use the built-in *File > Export > FBX* menu. Use **Check & Export** instead.

## Build, test, install

```bash
# From this folder:
blender --background --factory-startup --python tests/run_in_blender.py   # self-test, exits 0 if all pass
blender --command extension validate
blender --command extension build --output-dir ../dist
```

Install it from *Edit > Preferences > Get Extensions > ⌄ > Install from Disk…* by choosing the built `.zip`.

## Publish

Upload the built zip at <https://extensions.blender.org/submit/> after signing in with your Blender ID. Before you submit:

- Change `maintainer` in `blender_manifest.toml` if needed.
- Add a `website` field that points to your repository.
- Run `blender --command extension validate`.

License: GPL-3.0-or-later.
