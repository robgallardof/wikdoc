# Wikdoc usage examples

## Wizard (recommended)
```bash
wikdoc wizard
```

## Index a workspace
```bash
wikdoc index /path/to/workspace --local-store
```

## Ask questions
```bash
wikdoc ask /path/to/workspace "How does authentication work?" --local-store
```

## Generate wiki docs
```bash
wikdoc docs /path/to/workspace --out ./docs --template wiki --local-store
```

## Use global store (default)
```bash
wikdoc index /path/to/workspace
wikdoc ask /path/to/workspace "Where is X configured?"
```
