# Contributing to Mineral Exploration ML Pipeline

Thank you for your interest in contributing! This is an automated Paper2Code implementation that has been enhanced for usability and robustness. We welcome contributions to further improve it.

## Ways to Contribute

### 1. Bug Reports
If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Error messages and stack traces

### 2. Feature Requests
Have an idea for improvement? Open an issue describing:
- The feature or enhancement
- Use case and motivation
- Proposed implementation (if applicable)

### 3. Code Contributions
We welcome pull requests for:
- Bug fixes
- Performance improvements
- Better error handling
- Documentation improvements
- Additional features that align with the paper's methodology

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Enabling_Scalable_Mineral_Exploration_Self-Supervision_and_Explainability.git
cd Enabling_Scalable_Mineral_Exploration_Self-Supervision_and_Explainability
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Test that everything works:
```bash
python demo.py
```

## Making Changes

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following these guidelines:
   - Keep changes focused and minimal
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed
   - Test your changes with `python demo.py`

3. Commit your changes:
```bash
git add .
git commit -m "Brief description of your changes"
```

4. Push to your fork:
```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request on GitHub

## Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Write descriptive variable names
- Keep functions focused and modular
- Maximum line length: 120 characters (flexible for readability)

### Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md for user-facing changes
- Add inline comments for complex logic

### Error Handling
- Use try-except blocks for error-prone operations
- Provide helpful error messages
- Log errors appropriately
- Validate inputs early

## Testing

Before submitting a PR:

1. Run the demo script successfully:
```bash
python demo.py
```

2. Test with different config parameters if your changes affect configuration

3. If you have real geospatial data, test with it to ensure compatibility

## Areas for Improvement

Contributions are particularly welcome in these areas:

### High Priority
- [ ] Add unit tests for core functions
- [ ] Improve memory efficiency for large rasters
- [ ] Add progress bars for long-running operations
- [ ] Better visualization tools for predictions and attributions
- [ ] Support for additional raster formats
- [ ] Parallel processing for patch creation
- [ ] Model checkpointing during training
- [ ] TensorBoard integration for monitoring

### Medium Priority
- [ ] Command-line interface improvements
- [ ] Configuration validation
- [ ] Better handling of class imbalance
- [ ] Cross-validation support
- [ ] Hyperparameter tuning utilities
- [ ] Export results to common GIS formats

### Documentation
- [ ] Jupyter notebook tutorials
- [ ] Video walkthrough
- [ ] Example datasets
- [ ] API documentation
- [ ] Case studies with real data

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Check existing issues and PRs for similar topics
- Refer to the original paper for methodology questions

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to learn and improve this tool together.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

All contributors will be acknowledged in the project. Thank you for helping make this tool better!
