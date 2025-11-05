# Simple Makefile for data generation and cleanup

# Override with: make data SAMPLES=2000 OUTPUT=data
PY ?= python3
SAMPLES ?= 1000
OUTPUT ?= data
CHIRPS ?= 128
SAMPLES_PER_CHIRP ?= 256

.PHONY: data data-simple clean

# Realistic, physics-based dataset
data:
	$(PY) data_generator.py \
		--samples $(SAMPLES) \
		--output $(OUTPUT) \
		--chirps $(CHIRPS) \
		--samples-per-chirp $(SAMPLES_PER_CHIRP)

# Sanity-check dataset with distinct patterns (for debugging only)
data-simple:
	$(PY) dev/simple_data_generator.py \
		--samples $(SAMPLES) \
		--output $(OUTPUT) \
		--chirps $(CHIRPS) \
		--samples-per-chirp $(SAMPLES_PER_CHIRP)

# Cleanup artifacts (keeps best/final models by default)
clean:
	chmod +x clean.sh && ./clean.sh
