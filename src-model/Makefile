# Makefile for BGT60TR13C Finger Classification Project
# Full workflow: data generation, training, evaluation, visualization, cleanup

# Python interpreter (defaults to system python3, override with PY=/path/to/python)
PY ?= python3
VENV_PY ?= ../venv/bin/python

# Data generation parameters
SAMPLES ?= 1000
OUTPUT ?= data
CHIRPS ?= 128
SAMPLES_PER_CHIRP ?= 256

# Training parameters
EPOCHS ?= 35
BATCH_SIZE ?= 64
CONFIG ?= config.yaml

# Model and data paths
MODEL ?= models/best_model.pth
TRAIN_DATA ?= data/train_data.npz
VAL_DATA ?= data/val_data.npz

.PHONY: help all data data-simple train evaluate visualize clean clean-all

# Default target
help:
	@echo "BGT60TR13C Finger Classification - Makefile Targets"
	@echo "===================================================="
	@echo ""
	@echo "Data Generation:"
	@echo "  data          - Generate realistic training data (physics-based)"
	@echo "  data-simple   - Generate simple debugging dataset"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  train         - Train the model"
	@echo "  evaluate      - Evaluate trained model"
	@echo "  visualize     - Generate BGT60 simulation visualizations"
	@echo ""
	@echo "Workflow:"
	@echo "  all           - Run full pipeline: data + train + evaluate"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         - Remove logs, checkpoints, __pycache__ (keep results)"
	@echo "  clean-all     - Deep clean including models and datasets"
	@echo ""
	@echo "Parameters (override with VAR=value):"
	@echo "  SAMPLES=$(SAMPLES)  EPOCHS=$(EPOCHS)  BATCH_SIZE=$(BATCH_SIZE)"
	@echo "  PY=$(PY)  CONFIG=$(CONFIG)"

# Full pipeline
all: data train evaluate

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

# Train the model
train:
	@echo "Starting training..."
	$(PY) train.py \
		--config $(CONFIG) \
		--train-data $(TRAIN_DATA) \
		--val-data $(VAL_DATA) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE)

# Evaluate the trained model
evaluate:
	@echo "Evaluating model..."
	MPLBACKEND=Agg $(PY) evaluate.py \
		--config $(CONFIG) \
		--model $(MODEL) \
		--test-data $(VAL_DATA) \
		--output results

# Generate BGT60 simulation visualizations
visualize:
	@echo "Generating visualizations..."
	MPLBACKEND=Agg $(PY) visualize_bgt60.py

# Cleanup artifacts (keeps best/final models and results by default)
clean:
	@echo "Cleaning workspace (preserving results)..."
	@chmod +x clean.sh && ./clean.sh --no-results

# Deep clean including models and datasets
clean-all:
	@echo "Deep cleaning workspace..."
	@chmod +x clean.sh && printf 'y\n' | ./clean.sh --all-models --data
