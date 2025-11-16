#!/usr/bin/env python3
"""
Image Effects Workshop
Transform high-resolution Blender renders with various artistic effects.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import random
from scipy.ndimage import gaussian_filter


class ImageEffects:
    """Apply various artistic effects to images."""

    def __init__(self, image_path):
        """Initialize with an image path."""
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image = Image.open(self.image_path)
        self.image = self.image.convert('RGB')
        print(f"Loaded image: {self.image_path.name} ({self.image.size[0]}x{self.image.size[1]})")

    def paint_stroke_effect(self, stroke_size=8, stroke_length=12, stroke_width=3,
                           angle_variation=30, density=1.0):
        """
        Transform image into paint strokes by grouping pixels.

        Args:
            stroke_size: Size of pixel groups to sample (default: 8)
            stroke_length: Length of each paint stroke in pixels (default: 12)
            stroke_width: Width of each stroke (default: 3)
            angle_variation: Random angle variation in degrees (default: 30)
            density: Stroke density multiplier, 1.0 = full coverage (default: 1.0)

        Returns:
            PIL Image with paint stroke effect applied
        """
        print(f"Applying paint stroke effect (stroke_size={stroke_size}, "
              f"length={stroke_length}, width={stroke_width})...")

        width, height = self.image.size
        img_array = np.array(self.image)

        # Create a blank canvas (white background for paint effect)
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)

        # Calculate number of strokes based on density
        total_strokes = int((width * height) / (stroke_size ** 2) * density)

        # Generate strokes
        strokes = []
        for y in range(0, height, stroke_size):
            for x in range(0, width, stroke_size):
                # Sample the average color in this pixel group
                x_end = min(x + stroke_size, width)
                y_end = min(y + stroke_size, height)

                region = img_array[y:y_end, x:x_end]
                avg_color = region.mean(axis=(0, 1)).astype(int)
                color = tuple(avg_color)

                # Calculate stroke position (center of the region)
                cx = x + stroke_size // 2
                cy = y + stroke_size // 2

                # Random angle for more natural look
                angle = random.uniform(-angle_variation, angle_variation)
                angle_rad = np.radians(angle)

                # Calculate stroke endpoints
                dx = np.cos(angle_rad) * stroke_length / 2
                dy = np.sin(angle_rad) * stroke_length / 2

                x1 = cx - dx
                y1 = cy - dy
                x2 = cx + dx
                y2 = cy + dy

                strokes.append((x1, y1, x2, y2, color))

        # Shuffle strokes for more natural layering
        random.shuffle(strokes)

        # Draw strokes
        for i, (x1, y1, x2, y2, color) in enumerate(strokes):
            if i % 10000 == 0:
                print(f"  Drawing strokes... {i}/{len(strokes)}")
            draw.line([(x1, y1), (x2, y2)], fill=color, width=stroke_width)

        print("Paint stroke effect complete!")
        return canvas

    def japanese_calligraphy_effect(self, brush_size=12, stroke_length=25, tail_taper=0.7,
                                   edge_influence=0.7, density=1.2, follow_edges=True, ink_bleed=True,
                                   smart_tree_mode=False, directional_mode=False, directional_regions=None,
                                   variable_brush_size=False, min_brush_size=6, max_brush_size=18, size_variation=0.7):
        """
        Transform image using Japanese calligraphy-style brush strokes with edge detection.

        Args:
            brush_size: Base size of the brush (default: 12)
            stroke_length: Length of each calligraphy stroke (default: 25)
            tail_taper: How much the stroke tapers at the tail, 0-1 (default: 0.7)
            edge_influence: How strongly strokes follow edges, 0-1 (default: 0.7)
            density: Stroke density multiplier (default: 1.2)
            follow_edges: Whether to align strokes with detected edges (default: True)
            ink_bleed: Add subtle ink bleed effect (default: True)
            smart_tree_mode: Detect tree structures for directional bark strokes (default: False)

        Returns:
            PIL Image with Japanese calligraphy effect applied
        """
        print(f"Applying Japanese calligraphy effect (brush={brush_size}, "
              f"length={stroke_length}, edge_influence={edge_influence})...")

        width, height = self.image.size
        img_array = np.array(self.image)

        # Detect edges using Canny edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge gradient direction for stroke orientation
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_angles = np.arctan2(sobely, sobelx)

        # Directional mode: user-defined regions with custom stroke angles
        directional_mask = None
        directional_angle_map = None
        if directional_mode and directional_regions:
            print("  Processing directional regions...")
            directional_mask = np.zeros((height, width), dtype=np.uint8)
            directional_angle_map = np.zeros((height, width), dtype=np.float32)

            for region in directional_regions:
                if len(region.get('path', [])) < 3:
                    continue

                # Convert normalized coordinates to pixel coordinates
                path_pixels = [(int(p['x'] * width), int(p['y'] * height)) for p in region['path']]

                # Create mask for this region
                region_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(region_mask, [np.array(path_pixels, dtype=np.int32)], 255)

                # Set the angle for this region (convert degrees to radians)
                angle_rad = np.radians(region['angle'])

                # Update the directional mask and angle map
                directional_mask = cv2.bitwise_or(directional_mask, region_mask)
                directional_angle_map[region_mask > 0] = angle_rad

        # Smart tree mode: detect bark/trunk vs foliage
        bark_mask = None
        if smart_tree_mode:
            print("  Detecting tree structures (bark, trunks, branches)...")
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Detect brownish/grayish bark colors (tree trunks and branches)
            # Brown hues: 10-30 in OpenCV (0-180 scale)
            lower_brown = np.array([5, 20, 20])
            upper_brown = np.array([30, 255, 200])
            brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

            # Also detect gray/dark bark
            lower_gray = np.array([0, 0, 30])
            upper_gray = np.array([180, 50, 120])
            gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

            # Combine masks
            bark_mask = cv2.bitwise_or(brown_mask, gray_mask)

            # Morphological operations to clean up and connect bark regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            bark_mask = cv2.morphologyEx(bark_mask, cv2.MORPH_CLOSE, kernel)
            bark_mask = cv2.morphologyEx(bark_mask, cv2.MORPH_OPEN, kernel)

            # Dilate slightly to capture bark edges
            bark_mask = cv2.dilate(bark_mask, kernel, iterations=1)

        # Create canvas with off-white background (like rice paper)
        canvas = Image.new('RGB', (width, height), (248, 246, 240))
        draw = ImageDraw.Draw(canvas, 'RGBA')

        # Generate stroke positions
        strokes = []
        step_size = int(brush_size * 0.6)

        for y in range(0, height, step_size):
            for x in range(0, width, step_size):
                # Skip some strokes based on density
                if random.random() > density:
                    continue

                # Sample color from the region
                x_end = min(x + step_size, width)
                y_end = min(y + step_size, height)

                region = img_array[y:y_end, x:x_end]
                avg_color = region.mean(axis=(0, 1)).astype(int)

                # Add slight color variation for organic feel
                color_var = np.random.randint(-5, 6, 3)
                color = tuple(np.clip(avg_color + color_var, 0, 255))

                # Check if this pixel is in a directional region or bark region
                is_directional = False
                is_bark = False

                if directional_mode and directional_mask is not None:
                    is_directional = directional_mask[y, x] > 0

                if smart_tree_mode and bark_mask is not None:
                    is_bark = bark_mask[y, x] > 0

                # Determine stroke angle and characteristics
                if is_directional:
                    # DIRECTIONAL STROKES: User-defined regions with custom angles
                    angle = directional_angle_map[y, x]
                    # Add slight variation for natural look
                    angle += random.uniform(-0.15, 0.15)
                    length = stroke_length * random.uniform(1.5, 2.2)
                    width_mult = random.uniform(1.1, 1.4)
                    is_edge = False

                elif is_bark:
                    # BARK STROKES: Long, directional strokes following trunk/branch direction
                    # Calculate primary flow direction in this bark region
                    # Use gradient to determine bark growth direction (vertical for trunks, angled for branches)
                    flow_x = sobelx[y, x]
                    flow_y = sobely[y, x]
                    flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)

                    if flow_magnitude > 10:
                        # Follow the bark structure
                        angle = np.arctan2(flow_y, flow_x)
                    else:
                        # Default to vertical for trunk-like structures
                        angle = np.pi/2 + random.uniform(-0.3, 0.3)

                    # Longer, thicker strokes for bark texture
                    length = stroke_length * random.uniform(2.0, 2.8)
                    width_mult = random.uniform(1.2, 1.6)
                    is_edge = False

                elif follow_edges and edges[y, x] > 100:
                    # EDGE STROKES: Follow edges (for foliage details)
                    angle = edge_angles[y, x] + np.pi/2
                    angle += random.uniform(-0.2, 0.2)
                    length = stroke_length * 1.3
                    width_mult = 0.8
                    is_edge = True
                else:
                    # REGULAR STROKES: Random for foliage/background
                    angle = random.uniform(0, 2 * np.pi)
                    length = stroke_length
                    width_mult = 1.0
                    is_edge = False

                # Calculate stroke path with taper
                cx = x + step_size // 2
                cy = y + step_size // 2

                # Calculate brush size with optional variation
                if variable_brush_size:
                    # Map from detail to brush size
                    # More detail (higher edge value) = smaller brush
                    # Less detail = larger brush
                    detail_factor = edges[y, x] / 255.0  # Normalize edge strength

                    # size_variation controls how much detail affects size
                    # 1.0 = full range from min to max based on detail
                    # 0.0 = always use average size
                    base_size = (min_brush_size + max_brush_size) / 2
                    size_range = (max_brush_size - min_brush_size) / 2

                    # Invert detail factor so high detail = small brush
                    size_offset = (1.0 - detail_factor) * size_range * size_variation
                    current_brush_size = base_size + size_offset

                    # Add some randomness
                    current_brush_size *= random.uniform(0.9, 1.1)
                    current_brush_size = np.clip(current_brush_size, min_brush_size, max_brush_size)
                else:
                    current_brush_size = brush_size

                strokes.append({
                    'cx': cx,
                    'cy': cy,
                    'angle': angle,
                    'length': length,
                    'color': color,
                    'brush_size': current_brush_size * width_mult,
                    'tail_taper': tail_taper,
                    'is_edge': is_edge
                })

        # Sort strokes: non-edge strokes first, then edge strokes
        strokes.sort(key=lambda s: s['is_edge'])

        print(f"  Drawing {len(strokes)} calligraphy strokes...")

        # Draw strokes
        for i, stroke in enumerate(strokes):
            if i % 5000 == 0:
                print(f"  Progress... {i}/{len(strokes)}")

            self._draw_calligraphy_stroke(draw, stroke, ink_bleed)

        print("Japanese calligraphy effect complete!")
        return canvas

    def _draw_calligraphy_stroke(self, draw, stroke, ink_bleed):
        """Draw a single calligraphy brush stroke with taper and texture."""
        cx, cy = stroke['cx'], stroke['cy']
        angle = stroke['angle']
        length = stroke['length']
        color = stroke['color']
        brush_size = stroke['brush_size']
        tail_taper = stroke['tail_taper']

        # Number of segments for smooth curve
        segments = int(length / 3)
        if segments < 3:
            segments = 3

        # Add subtle curve to stroke
        curve_amount = random.uniform(-0.15, 0.15)

        for i in range(segments):
            t = i / segments

            # Calculate position along stroke
            progress = t
            curve_offset = np.sin(t * np.pi) * curve_amount * length / 4

            # Position
            base_x = cx + np.cos(angle) * length * (t - 0.5)
            base_y = cy + np.sin(angle) * length * (t - 0.5)

            # Add curve perpendicular to stroke direction
            x = base_x + np.cos(angle + np.pi/2) * curve_offset
            y = base_y + np.sin(angle + np.pi/2) * curve_offset

            # Calculate width with taper (wider in middle, thinner at ends)
            if t < 0.2:
                # Gradual start
                width_factor = t / 0.2 * 0.9
            elif t > 0.6:
                # Tapered tail
                tail_progress = (t - 0.6) / 0.4
                width_factor = 1.0 - (tail_progress * tail_taper)
            else:
                # Middle section
                width_factor = 0.9 + 0.1 * np.sin((t - 0.2) * np.pi / 0.4)

            current_width = max(1, int(brush_size * width_factor))

            # Color with alpha for ink effect
            alpha = int(220 + random.randint(-10, 10))
            if ink_bleed and t > 0.7:
                # More transparent at tail for bleed effect
                alpha = int(alpha * (1 - (t - 0.7) / 0.3 * 0.4))

            color_with_alpha = color + (alpha,)

            # Draw segment
            if i > 0:
                draw.line([(prev_x, prev_y), (x, y)],
                         fill=color_with_alpha, width=current_width)
            else:
                # Draw initial point
                draw.ellipse([x - current_width/2, y - current_width/2,
                            x + current_width/2, y + current_width/2],
                           fill=color_with_alpha)

            prev_x, prev_y = x, y

    def sumi_e_brush_effect(self, brush_size=15, min_stroke_length=20, max_stroke_length=60,
                           pressure_variation=0.8, texture_intensity=0.6, ink_dispersion=0.4,
                           flow_coherence=0.75, paper_texture=True):
        """
        Advanced Japanese Sumi-e brush painting with realistic texture and intentional flow.

        Creates layered, textured brush strokes that follow the image structure with
        natural ink behavior, bristle texture, and intentional artistic flow.

        Args:
            brush_size: Base size of brush bristles (default: 15)
            min_stroke_length: Minimum stroke length (default: 20)
            max_stroke_length: Maximum stroke length (default: 60)
            pressure_variation: Variation in brush pressure 0-1 (default: 0.8)
            texture_intensity: Amount of bristle texture 0-1 (default: 0.6)
            ink_dispersion: Ink bleeding/pooling effect 0-1 (default: 0.4)
            flow_coherence: How strokes follow image flow 0-1 (default: 0.75)
            paper_texture: Add subtle paper grain texture (default: True)

        Returns:
            PIL Image with Sumi-e brush effect applied
        """
        print(f"Applying Sumi-e brush effect (brush={brush_size}, "
              f"texture={texture_intensity}, flow={flow_coherence})...")

        width, height = self.image.size
        img_array = np.array(self.image)

        # Edge detection for structural awareness
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        # Calculate flow field (gradient direction)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        flow_angles = np.arctan2(sobely, sobelx)

        # Structure tensor for coherent flow
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        sobelx_blur = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
        sobely_blur = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

        # Create textured canvas (traditional Washi paper)
        if paper_texture:
            # Paper grain texture
            paper_noise = np.random.normal(0, 3, (height, width)).astype(np.float32)
            paper_grain = gaussian_filter(paper_noise, sigma=1.5)
            base_color = 252
            canvas_array = np.clip(base_color + paper_grain, 245, 255).astype(np.uint8)
            canvas_array = np.stack([canvas_array] * 3, axis=-1)
            canvas = Image.fromarray(canvas_array)
        else:
            canvas = Image.new('RGB', (width, height), (252, 250, 245))

        # Create alpha overlay for ink accumulation
        ink_layer = np.zeros((height, width, 4), dtype=np.float32)

        # Generate stroke seed points with smart sampling
        stroke_seeds = []
        step = int(brush_size * 0.5)

        for y in range(step, height - step, step):
            for x in range(step, width - step, step):
                # Sample image characteristics
                local_region = gray[max(0, y-step):min(height, y+step),
                                   max(0, x-step):min(width, x+step)]
                variance = np.var(local_region)

                # More strokes in detailed areas
                if variance > 100 or edges[y, x] > 50:
                    probability = 0.95
                else:
                    probability = 0.6

                if random.random() < probability:
                    stroke_seeds.append((x, y))

        print(f"  Generated {len(stroke_seeds)} stroke seeds...")

        # Create strokes with intentional flow
        strokes = []
        for seed_x, seed_y in stroke_seeds:
            # Determine stroke characteristics
            is_edge = edges[seed_y, seed_x] > 50
            base_angle = flow_angles[seed_y, seed_x]

            # Edge strokes are more precise
            if is_edge:
                angle = base_angle + np.pi/2 + random.uniform(-0.1, 0.1)
                stroke_length = random.uniform(min_stroke_length * 1.2, max_stroke_length * 0.8)
                width_mult = 0.7
                pressure = 0.85 + random.uniform(0, 0.15)
            else:
                # Flow strokes follow coherent direction
                angle_influence = flow_coherence
                random_angle = random.uniform(0, 2 * np.pi)
                angle = base_angle * angle_influence + random_angle * (1 - angle_influence)
                stroke_length = random.uniform(min_stroke_length, max_stroke_length)
                width_mult = 0.9 + random.uniform(-0.2, 0.3)
                pressure = 0.6 + random.uniform(0, 0.4)

            # Sample color from region
            region = img_array[max(0, seed_y-3):min(height, seed_y+3),
                             max(0, seed_x-3):min(width, seed_x+3)]
            if region.size > 0:
                avg_color = region.mean(axis=(0, 1))
                # Darken for ink effect
                ink_color = (avg_color * 0.85).astype(int)
                ink_color = tuple(np.clip(ink_color, 0, 255))
            else:
                ink_color = (60, 60, 60)

            strokes.append({
                'x': seed_x,
                'y': seed_y,
                'angle': angle,
                'length': stroke_length,
                'color': ink_color,
                'brush_size': brush_size * width_mult,
                'pressure': pressure * pressure_variation,
                'is_edge': is_edge,
                'texture': texture_intensity
            })

        # Sort: background strokes first, then details
        strokes.sort(key=lambda s: (s['is_edge'], -s['length']))

        print(f"  Painting {len(strokes)} Sumi-e strokes...")

        # Draw strokes with texture
        for i, stroke in enumerate(strokes):
            if i % 3000 == 0:
                print(f"    Progress... {i}/{len(strokes)}")

            self._draw_sumi_e_stroke(ink_layer, stroke, texture_intensity, ink_dispersion)

        # Composite ink layer onto canvas
        print("  Compositing ink layers...")
        canvas_array = np.array(canvas).astype(np.float32)

        for y in range(height):
            for x in range(width):
                alpha = ink_layer[y, x, 3] / 255.0
                if alpha > 0:
                    ink_rgb = ink_layer[y, x, :3]
                    canvas_array[y, x] = canvas_array[y, x] * (1 - alpha) + ink_rgb * alpha

        # Add subtle ink wash variations
        if ink_dispersion > 0.3:
            wash = gaussian_filter(ink_layer[:, :, 3], sigma=2.0)
            wash_normalized = (wash / wash.max() * 15 * ink_dispersion).astype(np.uint8)
            for c in range(3):
                canvas_array[:, :, c] = np.clip(canvas_array[:, :, c] - wash_normalized, 0, 255)

        result = Image.fromarray(canvas_array.astype(np.uint8))

        print("Sumi-e brush effect complete!")
        return result

    def _draw_sumi_e_stroke(self, ink_layer, stroke, texture_intensity, ink_dispersion):
        """Draw a single Sumi-e brush stroke with realistic texture."""
        x, y = stroke['x'], stroke['y']
        angle = stroke['angle']
        length = stroke['length']
        color = stroke['color']
        brush_size = stroke['brush_size']
        pressure = stroke['pressure']

        # Number of segments for smooth stroke
        segments = max(int(length / 2), 10)

        # Bristle simulation
        num_bristles = max(int(brush_size * texture_intensity * 2), 5)

        # Create stroke path with natural curve
        curve_amount = random.uniform(-0.2, 0.2)
        twist = random.uniform(-0.3, 0.3)

        for seg in range(segments):
            t = seg / segments

            # Position along stroke
            progress = t
            curve = np.sin(t * np.pi) * curve_amount * length / 3
            angle_offset = twist * np.sin(t * np.pi)

            px = x + np.cos(angle + angle_offset) * length * (t - 0.5)
            py = y + np.sin(angle + angle_offset) * length * (t - 0.5)
            px += np.cos(angle + angle_offset + np.pi/2) * curve
            py += np.sin(angle + angle_offset + np.pi/2) * curve

            # Pressure variation along stroke
            if t < 0.15:
                pressure_factor = (t / 0.15) * 0.7
            elif t > 0.75:
                pressure_factor = 1.0 - ((t - 0.75) / 0.25) * 0.85
            else:
                pressure_factor = 0.7 + 0.3 * np.sin((t - 0.15) / 0.6 * np.pi)

            current_pressure = pressure * pressure_factor
            current_width = brush_size * current_pressure

            # Draw bristles for texture
            for _ in range(num_bristles):
                bristle_offset_x = random.gauss(0, current_width * 0.3)
                bristle_offset_y = random.gauss(0, current_width * 0.3)

                bx = int(px + bristle_offset_x)
                by = int(py + bristle_offset_y)

                if 0 <= bx < ink_layer.shape[1] and 0 <= by < ink_layer.shape[0]:
                    # Ink intensity with variation
                    ink_alpha = current_pressure * 180 + random.randint(-20, 20)
                    ink_alpha = np.clip(ink_alpha, 0, 255)

                    # Color variation (bristle dryness)
                    color_var = random.randint(-8, 8)
                    bristle_color = tuple(np.clip(np.array(color) + color_var, 0, 255))

                    # Accumulate ink (additive blending with saturation)
                    existing_alpha = ink_layer[by, bx, 3]
                    new_alpha = min(255, existing_alpha + ink_alpha * 0.4)

                    blend = ink_alpha / 255.0
                    for c in range(3):
                        ink_layer[by, bx, c] = ink_layer[by, bx, c] * (1 - blend) + bristle_color[c] * blend

                    ink_layer[by, bx, 3] = new_alpha

            # Ink pooling/dispersion at stroke end
            if ink_dispersion > 0.2 and t > 0.8:
                pool_radius = int(current_width * ink_dispersion * 1.5)
                pool_x, pool_y = int(px), int(py)

                for dy in range(-pool_radius, pool_radius + 1):
                    for dx in range(-pool_radius, pool_radius + 1):
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < pool_radius:
                            pool_bx = pool_x + dx
                            pool_by = pool_y + dy

                            if 0 <= pool_bx < ink_layer.shape[1] and 0 <= pool_by < ink_layer.shape[0]:
                                pool_alpha = int((1 - dist/pool_radius) * 40 * ink_dispersion)
                                ink_layer[pool_by, pool_bx, 3] = min(255, ink_layer[pool_by, pool_bx, 3] + pool_alpha)

    def oil_paint_effect(self, min_brush_size=8, max_brush_size=25, stroke_length_mult=2.5,
                        impasto_strength=0.7, color_blend=0.5, detail_sensitivity=0.6,
                        stroke_direction_coherence=0.8, paint_thickness=0.6, fast_mode=False):
        """
        Thick oil paint effect with organic brush strokes and natural blending.

        Creates textured, layered brush strokes that adapt to image detail - larger bold
        strokes in uniform areas, smaller detailed strokes in complex regions. Strokes
        blend naturally when close together like wet paint.

        Args:
            min_brush_size: Minimum brush width in pixels (default: 8)
            max_brush_size: Maximum brush width in pixels (default: 25)
            stroke_length_mult: Stroke length multiplier relative to brush size (default: 2.5)
            impasto_strength: Thickness/texture of paint application 0-1 (default: 0.7)
            color_blend: How much adjacent strokes blend 0-1 (default: 0.5)
            detail_sensitivity: How responsive to fine details 0-1 (default: 0.6)
            stroke_direction_coherence: Stroke flow following image 0-1 (default: 0.8)
            paint_thickness: Visual paint buildup effect 0-1 (default: 0.6)
            fast_mode: Use simplified mode for faster processing (default: False)

        Returns:
            PIL Image with oil paint effect applied
        """
        mode_text = "FAST" if fast_mode else "QUALITY"
        print(f"Applying oil paint effect [{mode_text}] (brush={min_brush_size}-{max_brush_size}, "
              f"impasto={impasto_strength}, detail={detail_sensitivity})...")

        width, height = self.image.size
        img_array = np.array(self.image)

        # Analyze image structure
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Detail map - where to use fine vs broad strokes
        detail_map = cv2.Laplacian(gray, cv2.CV_64F)
        detail_map = np.abs(detail_map)
        detail_map = cv2.GaussianBlur(detail_map, (5, 5), 0)
        detail_map = (detail_map / detail_map.max()) if detail_map.max() > 0 else detail_map

        # Flow field for stroke direction
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        flow_angles = np.arctan2(sobely, sobelx)
        flow_angles = cv2.GaussianBlur(flow_angles, (7, 7), 0)

        # Create canvas
        canvas = Image.new('RGB', (width, height), 'white')
        paint_layer = np.array(canvas).astype(np.float32)

        # Paint thickness accumulation layer
        thickness_layer = np.zeros((height, width), dtype=np.float32)

        # Generate stroke positions adaptively
        stroke_data = []
        total_strokes_estimate = 0

        # Optimize for fast mode or large images
        if fast_mode or (width * height) > 5000000:  # 5MP+
            # Single pass, larger steps, fewer strokes
            passes = [
                {'min_detail': 0.0, 'max_detail': 1.0, 'size_bias': 0.6},
            ]
            density_mult = 0.4 if fast_mode else 0.6
            print("  Using optimized mode for faster processing...")
        else:
            # Multi-pass: broad strokes first, then details
            passes = [
                {'min_detail': 0.0, 'max_detail': 0.3, 'size_bias': 0.8},  # Large strokes
                {'min_detail': 0.2, 'max_detail': 0.6, 'size_bias': 0.5},  # Medium strokes
                {'min_detail': 0.5, 'max_detail': 1.0, 'size_bias': 0.2},  # Fine details
            ]
            density_mult = 1.0

        print("  Analyzing image and planning strokes...")

        for pass_idx, pass_config in enumerate(passes):
            min_d, max_d = pass_config['min_detail'], pass_config['max_detail']
            size_bias = pass_config['size_bias']

            # Adaptive sampling based on detail - MUCH larger steps
            base_step = int(min_brush_size * (2.5 - size_bias))

            for y in range(0, height, base_step):
                for x in range(0, width, base_step):
                    if y >= height or x >= width:
                        continue

                    detail_level = detail_map[min(y, height-1), min(x, width-1)]

                    # Skip if detail doesn't match this pass
                    if detail_level < min_d or detail_level > max_d:
                        continue

                    # Adaptive sampling density with optimization
                    density_threshold = (0.5 + detail_level * detail_sensitivity * 0.3) * density_mult
                    if random.random() > density_threshold:
                        continue

                    # Determine brush size based on detail
                    detail_factor = 1.0 - (detail_level * detail_sensitivity)
                    brush_size = min_brush_size + (max_brush_size - min_brush_size) * detail_factor * size_bias

                    # Sample color
                    sample_radius = max(1, int(brush_size * 0.4))
                    y_start = max(0, y - sample_radius)
                    y_end = min(height, y + sample_radius)
                    x_start = max(0, x - sample_radius)
                    x_end = min(width, x + sample_radius)

                    region = img_array[y_start:y_end, x_start:x_end]
                    if region.size > 0:
                        color = region.mean(axis=(0, 1))
                        color = tuple(np.clip(color, 0, 255).astype(int))
                    else:
                        continue

                    # Stroke direction
                    base_angle = flow_angles[min(y, height-1), min(x, width-1)]
                    if stroke_direction_coherence < 1.0:
                        random_component = random.uniform(0, 2 * np.pi)
                        angle = base_angle * stroke_direction_coherence + random_component * (1 - stroke_direction_coherence)
                    else:
                        angle = base_angle

                    angle += random.uniform(-0.2, 0.2)  # Small variation

                    # Stroke length
                    stroke_length = brush_size * stroke_length_mult * random.uniform(0.8, 1.2)

                    stroke_data.append({
                        'x': x,
                        'y': y,
                        'angle': angle,
                        'length': stroke_length,
                        'width': brush_size,
                        'color': color,
                        'pass': pass_idx,
                        'detail': detail_level
                    })

        total_strokes_estimate = len(stroke_data)
        print(f"  Generated {total_strokes_estimate} strokes across {len(passes)} passes...")

        # Paint strokes
        for idx, stroke in enumerate(stroke_data):
            if idx % 2000 == 0:
                progress = int((idx / total_strokes_estimate) * 100)
                print(f"  Progress: {progress}% ({idx}/{total_strokes_estimate})")

            self._draw_oil_stroke(paint_layer, thickness_layer, stroke,
                                 impasto_strength, color_blend, paint_thickness)

        # Apply slight color variation for organic feel
        if impasto_strength > 0.5:
            noise = np.random.normal(0, 2, paint_layer.shape).astype(np.float32)
            paint_layer = np.clip(paint_layer + noise * impasto_strength, 0, 255)

        result = Image.fromarray(paint_layer.astype(np.uint8))

        print("Oil paint effect complete!")
        return result

    def _draw_oil_stroke(self, canvas, thickness_layer, stroke, impasto, color_blend, paint_thickness):
        """Draw a single oil paint brush stroke with blending."""
        x, y = stroke['x'], stroke['y']
        angle = stroke['angle']
        length = stroke['length']
        width = stroke['width']
        color = np.array(stroke['color'], dtype=np.float32)

        # Number of segments
        segments = max(int(length / 3), 5)

        # Stroke path with natural curve
        curve = random.uniform(-0.15, 0.15)

        for i in range(segments):
            t = i / max(segments - 1, 1)

            # Position along stroke
            curve_offset = np.sin(t * np.pi) * curve * length / 4

            sx = x + np.cos(angle) * length * (t - 0.5)
            sy = y + np.sin(angle) * length * (t - 0.5)
            sx += np.cos(angle + np.pi/2) * curve_offset
            sy += np.sin(angle + np.pi/2) * curve_offset

            # Width variation (thicker in middle)
            width_factor = 0.6 + 0.4 * np.sin(t * np.pi)
            current_width = width * width_factor * (0.9 + random.uniform(0, 0.2))

            # Paint this segment
            half_w = int(current_width / 2) + 1

            for dy in range(-half_w, half_w + 1):
                for dx in range(-half_w, half_w + 1):
                    dist = np.sqrt(dx**2 + dy**2)

                    if dist > current_width / 2:
                        continue

                    px = int(sx + dx)
                    py = int(sy + dy)

                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                        # Paint opacity based on distance from center
                        opacity = (1.0 - dist / (current_width / 2)) * impasto

                        # Blend with existing paint (wet-on-wet)
                        existing_thickness = thickness_layer[py, px]
                        blend_factor = color_blend * min(1.0, existing_thickness)

                        # Mix colors
                        existing_color = canvas[py, px]
                        blended_color = color * (1 - blend_factor) + existing_color * blend_factor

                        # Apply paint
                        canvas[py, px] = canvas[py, px] * (1 - opacity) + blended_color * opacity

                        # Accumulate thickness
                        thickness_layer[py, px] = min(1.0, existing_thickness + opacity * paint_thickness * 0.3)

    def linocut_effect(self, num_colors=4, color_boost=1.3, outline_thickness=2,
                      texture_intensity=0.6, carve_direction='mixed', contrast=1.2):
        """
        Linocut/woodblock print effect with bold shapes and limited color palette.

        Creates a graphic print aesthetic with posterized colors, carved texture,
        and optional bold outlines. High contrast, flat color areas like traditional
        relief printing.

        Args:
            num_colors: Number of colors in palette (2-8) (default: 4)
            color_boost: Saturation/vibrancy multiplier (default: 1.3)
            outline_thickness: Thickness of outlines in pixels, 0=none (default: 2)
            texture_intensity: Amount of carved texture/grain 0-1 (default: 0.6)
            carve_direction: Texture direction - 'horizontal', 'vertical', 'mixed' (default: 'mixed')
            contrast: Contrast boost for graphic look (default: 1.2)

        Returns:
            PIL Image with linocut effect applied
        """
        print(f"Applying linocut effect (colors={num_colors}, texture={texture_intensity}, "
              f"outlines={outline_thickness})...")

        width, height = self.image.size
        img_array = np.array(self.image).astype(np.float32)

        # Boost contrast for graphic look
        if contrast != 1.0:
            mean = img_array.mean()
            img_array = mean + (img_array - mean) * contrast
            img_array = np.clip(img_array, 0, 255)

        # Convert to LAB for better color quantization
        img_pil = Image.fromarray(img_array.astype(np.uint8))

        # Color quantization using k-means clustering
        print("  Posterizing colors...")
        pixels = img_array.reshape(-1, 3)

        # Use k-means to find dominant colors
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42, batch_size=1000)
        kmeans.fit(pixels)

        # Get palette
        palette = kmeans.cluster_centers_

        # Boost color saturation
        if color_boost != 1.0:
            for i in range(len(palette)):
                # Convert to HSV to boost saturation
                rgb = palette[i] / 255.0
                r, g, b = rgb[0], rgb[1], rgb[2]

                max_c = max(r, g, b)
                min_c = min(r, g, b)
                diff = max_c - min_c

                if diff > 0:
                    # Boost saturation
                    if max_c == r:
                        h = ((g - b) / diff) % 6
                    elif max_c == g:
                        h = ((b - r) / diff) + 2
                    else:
                        h = ((r - g) / diff) + 4

                    s = diff / max_c if max_c > 0 else 0
                    v = max_c

                    # Boost saturation
                    s = min(1.0, s * color_boost)

                    # Convert back to RGB
                    c = v * s
                    x = c * (1 - abs((h % 2) - 1))
                    m = v - c

                    if h < 1:
                        r, g, b = c, x, 0
                    elif h < 2:
                        r, g, b = x, c, 0
                    elif h < 3:
                        r, g, b = 0, c, x
                    elif h < 4:
                        r, g, b = 0, x, c
                    elif h < 5:
                        r, g, b = x, 0, c
                    else:
                        r, g, b = c, 0, x

                    palette[i] = np.array([r + m, g + m, b + m]) * 255

        # Assign each pixel to nearest palette color
        labels = kmeans.predict(pixels)
        posterized = palette[labels].reshape(height, width, 3).astype(np.uint8)

        # Detect edges for outlines
        if outline_thickness > 0:
            print("  Adding bold outlines...")
            gray = cv2.cvtColor(posterized, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges for thickness
            if outline_thickness > 1:
                kernel = np.ones((outline_thickness, outline_thickness), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=1)

            # Apply black outlines
            posterized[edges > 0] = [0, 0, 0]

        # Add carved texture/grain
        if texture_intensity > 0:
            print("  Carving texture...")
            texture = np.zeros((height, width), dtype=np.float32)

            if carve_direction == 'horizontal':
                # Horizontal carving lines
                for y in range(height):
                    noise = random.gauss(0, texture_intensity * 8)
                    texture[y, :] = noise + np.sin(y * 0.3) * texture_intensity * 5
            elif carve_direction == 'vertical':
                # Vertical carving lines
                for x in range(width):
                    noise = random.gauss(0, texture_intensity * 8)
                    texture[:, x] = noise + np.sin(x * 0.3) * texture_intensity * 5
            else:  # mixed
                # Mixed direction based on local structure
                for y in range(0, height, 4):
                    for x in range(0, width, 4):
                        direction = random.choice(['h', 'v', 'd'])
                        noise = random.gauss(0, texture_intensity * 6)

                        if direction == 'h':
                            texture[y:y+4, x:x+4] += noise
                        elif direction == 'v':
                            texture[y:y+4, x:x+4] += noise * 0.7
                        else:
                            # Diagonal
                            for i in range(4):
                                for j in range(4):
                                    if y+i < height and x+j < width:
                                        texture[y+i, x+j] += noise * (1 - abs(i-j)/4)

            # Apply texture with blur
            texture = cv2.GaussianBlur(texture, (3, 3), 0)

            # Add texture to image
            for c in range(3):
                posterized[:, :, c] = np.clip(posterized[:, :, c].astype(np.float32) + texture, 0, 255).astype(np.uint8)

        result = Image.fromarray(posterized)

        print("Linocut effect complete!")
        return result

    def save_result(self, processed_image, output_path=None, suffix="_effect"):
        """Save the processed image."""
        if output_path is None:
            output_path = self.image_path.parent / f"{self.image_path.stem}{suffix}{self.image_path.suffix}"
        else:
            output_path = Path(output_path)

        processed_image.save(output_path, quality=95)
        print(f"Saved result to: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Apply artistic effects to high-resolution images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paint stroke effect with default settings
  python image_effects.py input.png --effect paint_stroke

  # Customize paint stroke parameters
  python image_effects.py input.png --effect paint_stroke --stroke-size 10 --stroke-length 15 --stroke-width 4

  # Specify output path
  python image_effects.py input.png --effect paint_stroke --output result.png
        """
    )

    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--effect', type=str, required=True,
                       choices=['paint_stroke'],
                       help='Effect to apply')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: adds suffix to input filename)')

    # Paint stroke effect parameters
    paint_group = parser.add_argument_group('paint_stroke parameters')
    paint_group.add_argument('--stroke-size', type=int, default=8,
                            help='Size of pixel groups to sample (default: 8)')
    paint_group.add_argument('--stroke-length', type=int, default=12,
                            help='Length of each stroke (default: 12)')
    paint_group.add_argument('--stroke-width', type=int, default=3,
                            help='Width of each stroke (default: 3)')
    paint_group.add_argument('--angle-variation', type=float, default=30,
                            help='Random angle variation in degrees (default: 30)')
    paint_group.add_argument('--density', type=float, default=1.0,
                            help='Stroke density multiplier (default: 1.0)')

    args = parser.parse_args()

    try:
        # Load image
        effects = ImageEffects(args.image)

        # Apply effect
        if args.effect == 'paint_stroke':
            result = effects.paint_stroke_effect(
                stroke_size=args.stroke_size,
                stroke_length=args.stroke_length,
                stroke_width=args.stroke_width,
                angle_variation=args.angle_variation,
                density=args.density
            )

        # Save result
        effects.save_result(result, args.output, suffix=f"_{args.effect}")

        print("\nDone!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
