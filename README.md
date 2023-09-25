# NLP POS Tagging Project

## Team: 
- ZIDANI Yasser
- Thomas GAVIARD

This repository contains implementations of various classifiers to predict Part-of-Speech (POS) tags for French text.

## Classifiers

1. **NaiveClassifier**: A rule-based classifier.
2. **RandomClassifier**: Assigns POS tags at random.
3. **StratifiedClassifier**: Assigns POS tags based on the actual POS distribution.
4. **MostCommonClassifier**: Assigns the most common POS tag to each word.

## Results

- **NaiveClassifier**: 62.16%
- **RandomClassifier**:
  - Seed 0: 6.28%
  - Seed 1: 6.28%
  - Seed 2: 6.12%
  - Seed 3: 6.25%
  - Seed 4: 6.36%
  - Seed 5: 6.23%
  - Seed 6: 6.23%
  - Seed 7: 6.26%
  - Seed 8: 6.24%
  - Seed 9: 6.18%
- **StratifiedClassifier**: 11.66%
- **MostCommonClassifier**: 18.73%

## Getting Started

To run the classifiers, follow the instructions in the `CONTRIBUTING.md` file.

## License

This project is open-source and available under the MIT License.

