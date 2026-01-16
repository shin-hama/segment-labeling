# Copilot Instructions for Segment Labeling Project

## Project Overview

This is an image segmentation and labeling tool built with:

- **Segment Anything Model (SAM)** for image segmentation
- **Gradio** for web-based user interface
- **OpenCV** for image processing
- **PyTorch** for model inference

## Key Technologies and Libraries

- `segment_anything`: Meta's SAM model for zero-shot image segmentation
- `gradio`: Web UI framework
- `opencv-python`: Computer vision operations
- `torch`/`torchvision`: Deep learning framework
- `PIL`: Image processing
- `numpy`: Array operations

## Code Style Guidelines

### General Principles

- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines (enforced by ruff linter)
- Use descriptive variable names, especially for computer vision operations
- Add docstrings to all functions explaining their purpose and parameters

### Computer Vision Specific Guidelines

- Use `np.ndarray` type hints for image arrays
- Specify image format in comments (e.g., "RGB format", "BGR format", "grayscale")
- Always handle both CPU and GPU operations gracefully
- Use consistent coordinate systems (x, y vs row, col)

### SAM Model Integration

- Always check device availability before model operations
- Use the global `predictor` instance for inference
- Handle mask arrays with proper shape validation
- Convert between different mask formats (boolean, uint8) as needed

### Gradio Interface Guidelines

- Use clear component names and descriptions
- Implement proper state management for multi-step interactions
- Handle file uploads and downloads securely
- Provide user feedback for long-running operations

### Error Handling

- Validate image inputs (format, size, channels)
- Handle CUDA availability gracefully
- Provide meaningful error messages for user actions
- Log errors for debugging purposes

## Architecture Patterns

### State Management

The project uses a global `AnnotationState` class to manage:

- Current image being processed
- Generated masks and scores
- User annotations and labels
- Selected mask indices

When modifying state:

```python
# Always update state consistently
state.current_image = processed_image
state.current_masks = masks
state.current_scores = scores
```

### Image Processing Pipeline

1. Load and validate input image
2. Set image in SAM predictor
3. Generate segmentation masks
4. Process user interactions (clicks, labels)
5. Export annotations in desired format

### Function Organization

- **Initialization functions**: Setup SAM model and global state
- **Image processing functions**: Handle computer vision operations
- **UI callback functions**: Process Gradio interface interactions
- **Export functions**: Save annotations in various formats

## Performance Considerations

- Batch process multiple masks when possible
- Cache expensive computations (SAM embeddings)
- Use appropriate image resolutions for display vs processing
- Optimize mask operations for large images

## Common Patterns

### Working with SAM masks

```python
# Generate masks from point prompts
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)
```

### Image format conversions

```python
# PIL to numpy (RGB)
image_array = np.array(pil_image)

# OpenCV uses BGR, convert when needed
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
```

### Polygon extraction from masks

```python
# Convert binary mask to polygon coordinates
contours, _ = cv2.findContours(
    mask.astype(np.uint8),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

## Testing Guidelines

- Test with images of different sizes and formats
- Verify GPU/CPU compatibility
- Test edge cases (empty selections, invalid coordinates)
- Validate export formats match expected schemas

## File Naming Conventions

- Use snake_case for Python files and functions
- Use descriptive names for computer vision operations
- Prefix UI callback functions with appropriate action names
- Group related functions in logical sections

## Dependencies Management

- Use `pyproject.toml` for dependency specification
- Pin major versions for stability
- Document any special installation requirements
- Keep development dependencies separate

Remember: This is a research/prototype tool for computer vision annotation tasks. Prioritize code clarity and user experience over premature optimization.
