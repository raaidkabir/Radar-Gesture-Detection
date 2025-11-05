# Finger Classification with BGT60TR13C Radar

Physics-based simulator and ML pipeline for classifying fingers using the Infineon BGT60TR13C 60 GHz radar.

## What is this?

We're using a 60 GHz millimeter-wave radar to detect and classify individual fingers (thumb, index, middle, ring, pinky). This project simulates how the radar sees fingers and trains a model to identify them.

## Hardware: BGT60TR13C Specs

- **Frequency**: 58-63.5 GHz (center: 60.75 GHz) - that's *tiny* wavelengths (~5mm)!
- **Bandwidth**: 5.5 GHz → gives us ~2.7 cm range resolution
- **Channels**: 1 transmitter, 3 receivers
- **ADC**: 12-bit, 4 MSps sampling
- **Sweet spot**: Works best at 15-70 cm distance

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the simulator
```bash
python3 rcs_simulator.py
```

### 3. Generate training data
```bash
# Small test (100 samples per finger)
python3 data_generator.py --samples 100 --output data

# Full dataset (2000 samples per finger)
python3 data_generator.py --samples 2000 --output data
```

Or just run everything at once:
```bash
./quickstart.sh
```

If you want a sanity-check dataset with extremely distinct patterns (for pipeline validation), use the simple generator under dev/:

```bash
# Sanity dataset (fast, separable patterns)
python3 dev/simple_data_generator.py --samples 200 --output data
```
Use the realistic `data_generator.py` for any actual training runs you care about; the simple generator is only for debugging the training pipeline.

### Make targets

From `src-model/`, you can also use the Makefile shortcuts:

```bash
# Realistic dataset (override SAMPLES/OUTPUT/etc as needed)
make data SAMPLES=1000 OUTPUT=data

# Simple sanity dataset
make data-simple SAMPLES=200 OUTPUT=data

# Cleanup artifacts (keeps best/final models by default)
make clean
```

## How It Works

## How It Works

**The Physics:** We model fingers as cylinders and calculate how much radar energy bounces back (RCS). At 60 GHz, the radio waves only penetrate ~0.4 mm into skin, so we're mainly seeing surface reflections.

**The Data:** The simulator generates Range-Doppler maps (think: 2D heatmaps showing where things are and how fast they're moving). Each map is 128×256 pixels representing what the radar "sees."

**The Model:** A neural network learns to classify these Range-Doppler patterns into the 5 finger types.

## What Makes 60 GHz Special?

- ✅ **High resolution** - 5mm wavelength means we can see tiny details
- ✅ **Surface-only** - Shallow penetration (~0.4mm) = clean finger surface detection
- ✅ **Good reflectivity** - Human skin reflects ~46% of the signal
- ⚠️ **Short range** - Oxygen absorbs 60 GHz signals, limiting range to ~1m
- ⚠️ **Line-of-sight only** - Can't see through walls (which is actually good for privacy!)

## Detection Performance

| Distance | Thumb | Index | Middle | Ring | Pinky |
|----------|-------|-------|--------|------|-------|
| 0.2m     | ✓✓    | ✓✓    | ✓✓     | ✓✓   | ✓✓    |
| 0.5m     | ✓     | ✓     | ✓      | ✓    | ✓     |
| 1.0m     | ✗     | ✗     | ~      | ✗    | ✗     |

**Legend:** ✓✓ Excellent | ✓ Good | ~ Marginal | ✗ Poor

**TL;DR:** Keep your hand within 50cm of the radar for best results.

## Finger Dimensions & RCS

## Finger Dimensions & RCS

| Finger | Length | Diameter | RCS @ 0.2m |
|--------|--------|----------|------------|
| Thumb  | 48 mm  | 20 mm    | -6.6 dBsm  |
| Index  | 70 mm  | 16 mm    | -5.3 dBsm  |
| Middle | 76 mm  | 17 mm    | -4.3 dBsm  | ← Biggest!
| Ring   | 69 mm  | 16 mm    | -5.7 dBsm  |
| Pinky  | 58 mm  | 14 mm    | -8.4 dBsm  | ← Smallest!

The middle finger has the highest radar signature, pinky has the lowest.

## Files

- **`rcs_simulator.py`** - Physics simulator for 60 GHz radar + fingers
- **`data_generator.py`** - Generates synthetic training data (Range-Doppler maps)
- **`visualize_bgt60.py`** - Analysis tools and plots
- **`config.yaml`** - Hyperparameters
- **`requirements.txt`** - Python packages
- **`quickstart.sh`** - One-command setup

## Advanced: Gesture Sequences

Want to detect gestures (not just classify fingers)? The generator can create temporal sequences:

```python
from data_generator import BGT60DataGenerator

gen = BGT60DataGenerator()
frames, metadata = gen.generate_gesture_sequence('swipe_right', num_frames=50)
```

Gestures: `swipe_right`, `swipe_left`, `tap`, `wave`, `pinch`

## References & Notes

## References & Notes

- Hardware: [Infineon BGT60TR13C Datasheet](Radar_data_sheet.pdf)
- Human tissue properties at mmWave: ITU-R P.2040
- This is a *simulation* - real hardware will need calibration!

## TODO / Future Work

- [ ] Train actual model (`model.py`, `train.py`, `evaluate.py` needed)
- [ ] Test with real BGT60TR13C hardware
- [ ] Multi-finger tracking
- [ ] Real-time gesture recognition
- [ ] Transfer learning from sim → real data

## Questions?

Check `IMPLEMENTATION_SUMMARY.md` for technical deep-dive.

---

**Made for EE592A Project** 🎯

## Cleanup

Generated artifacts can pile up during experiments. Use the cleanup helper to tidy the workspace:

```bash
# From src-model/
chmod +x clean.sh
./clean.sh            # remove logs/, results/, __pycache__/, checkpoint_*.pth

./clean.sh --dry-run  # show what would be deleted
./clean.sh --data     # also remove data/*.npz (datasets)
./clean.sh --all-models   # remove ALL models/*.pth and models/*.npz

# Keep logs or results:
./clean.sh --no-logs --no-results
```

Git hygiene: a top-level `.gitignore` excludes caches, virtualenvs, logs, results, model weights, and datasets from version control by default.
