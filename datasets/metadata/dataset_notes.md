# Dataset Notes

## Sources

### DSBI (Dots Braille Image Database)
- **Format**: Images + `.+recto.txt` / `.+verso.txt` dot coordinate files
- **Dot format**: One dot per line, last two columns = x y pixel coordinates
- **Split files**: `train.txt` and `test.txt` map stems to splits
- **Website**: http://www.cs.ru.nl/~blagojevich/dsbi/
- **License**: Research use

### Angelina Dataset
- **Format**: Images + JSON annotation files per image
- **JSON structure**: `{"marks": [{"x", "y", "w", "h", "label"}, ...]}`
- **Subsets**: books, handwritten, pics, uploaded, not_braille
- **Website**: https://github.com/IlyaOvodov/AngelinaReader
- **License**: See Angelina project

## Bit Convention

All pattern integers use this bit mapping:

```
bit 0 = dot 1 (top-left)    bit 3 = dot 4 (top-right)
bit 1 = dot 2 (mid-left)    bit 4 = dot 5 (mid-right)
bit 2 = dot 3 (bot-left)    bit 5 = dot 6 (bot-right)
```

So `a` = 0b000001 = 1, `b` = 0b000011 = 3, etc.

## Split Strategy

- DSBI: uses train.txt/test.txt if present; every 5th test image → val
- Angelina: consecutive index split (not random) to keep physical pages together
- `uploaded` subset goes entirely to test (real-world evaluation)
- `not_braille` subset is train-only hard negatives (empty label files)

## Class Distribution Notes

- 65 CNN classes: 0–63 (6-bit patterns) + 64 (blank)
- Only ~29 of the 64 patterns appear in typical Grade 1 text
- Rare classes (z, x, q, w) need weighted sampling — see `generate_splits.py` output

## Preprocessing Applied at Crop Time

- Per-crop CLAHE: clipLimit=3.0, tileGridSize=(4,4)
- Output size: 64×64 grayscale PNG
- Padding: 10% on each side
