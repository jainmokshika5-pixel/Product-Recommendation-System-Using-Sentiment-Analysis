# Contributing to AI Product Recommendation System

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ai-product-recommendation.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit your changes: `git commit -m 'Add some feature'`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## 📋 Development Guidelines

### Code Style

**Python:**
- Follow PEP 8 style guide
- Use type hints where applicable
- Add docstrings to functions and classes
- Keep functions focused and small

**TypeScript/React:**
- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Keep components small and reusable

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 50 characters
- Add detailed description if needed

Examples:
```
Add sentiment analysis for product reviews
Fix bug in recommendation algorithm
Update README with installation instructions
```

## 🧪 Testing

Before submitting a PR:

1. **Test the backend:**
```bash
cd backend
python -m pytest tests/
```

2. **Test the frontend:**
```bash
npm run test
```

3. **Run the full application:**
```bash
npm run start:full
```

4. **Check for linting errors:**
```bash
npm run lint
```

## 📝 Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation if you're changing functionality
3. Ensure all tests pass
4. Make sure your code follows the style guidelines
5. Request review from maintainers

## 🐛 Reporting Bugs

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, Node version)

## 💡 Suggesting Features

We welcome feature suggestions! Please:

- Check if the feature has already been suggested
- Provide a clear description of the feature
- Explain why it would be useful
- Include examples if possible

## 🔍 Areas for Contribution

- **Model Improvements**: Better architectures, hyperparameter tuning
- **New Features**: Additional recommendation algorithms, user preferences
- **UI/UX**: Improved interface, better visualizations
- **Documentation**: Tutorials, examples, API documentation
- **Testing**: Unit tests, integration tests, end-to-end tests
- **Performance**: Optimization, caching, faster inference
- **Bug Fixes**: Fix reported issues

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)

## ❓ Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉
