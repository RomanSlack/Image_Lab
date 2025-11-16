# Image Effects Workshop

Transform high-resolution Blender renders with various artistic effects.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic syntax:
```bash
python image_effects.py <image_path> --effect <effect_name> [options]
```

### Paint Stroke Effect

Transform your image into a painting with colored brush strokes:

```bash
# Basic usage
python image_effects.py tree_render.png --effect paint_stroke

# Customize stroke parameters
python image_effects.py tree_render.png --effect paint_stroke \
    --stroke-size 10 \
    --stroke-length 15 \
    --stroke-width 4 \
    --angle-variation 45

# Specify output path
python image_effects.py tree_render.png --effect paint_stroke \
    --output my_painting.png
```

#### Parameters:
- `--stroke-size`: Size of pixel groups to sample (default: 8) - larger = more abstract
- `--stroke-length`: Length of each stroke in pixels (default: 12)
- `--stroke-width`: Width/thickness of strokes (default: 3)
- `--angle-variation`: Random angle variation in degrees (default: 30) - more = wilder strokes
- `--density`: Stroke coverage multiplier (default: 1.0) - lower = more sparse

## Adding New Effects

The script is designed to be easily extensible. To add a new effect:

1. Add a new method to the `ImageEffects` class in `image_effects.py`
2. Add the effect name to the `--effect` choices in the argument parser
3. Add any new parameters to an argument group
4. Add the effect case in the main function

Example structure:
```python
def your_new_effect(self, param1=10, param2=5):
    """Your effect description."""
    # Your effect code here
    return processed_image
```

## Output

By default, processed images are saved with the effect name as a suffix (e.g., `tree_render_paint_stroke.png`). Use `--output` to specify a custom path.
