# 🧬 COVID Drug Discovery ML Model

A machine learning project focused on **COVID-19 drug discovery simulation** using hybrid models and SPSA (Simultaneous Perturbation Stochastic Approximation) optimization.

## 📁 Project Structure

```
ml model/
├── train.py                  # Main training script
├── hybrid_model.py           # Hybrid ML model architecture
├── data_loader.py            # Data loading and preprocessing
├── covid_drug_sim.py         # COVID drug simulation (baseline)
├── covid_drug_sim_spsa.py    # COVID drug simulation with SPSA optimizer
├── drug_discovery_sim.py     # Drug discovery simulation
├── molecule_sim.py           # Molecule simulation utilities
├── spsa_results.txt          # SPSA optimization results
└── data/                     # Dataset directory
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/https://github.com/mohammadikhlakh56-hub/COVID-19-drug-discovery-simulation/ml-model.git
cd ml-model

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Model

```bash
# Train the hybrid model
python train.py

# Run COVID drug simulation
python covid_drug_sim.py

# Run with SPSA optimization
python covid_drug_sim_spsa.py
```

## 🧪 Methods

- **Hybrid Model**: Combines multiple ML techniques for drug-protein interaction prediction
- **SPSA Optimization**: Gradient-free optimizer for fine-tuning simulation parameters
- **Molecule Simulation**: Physics-inspired molecular property estimation

## 📊 Results

See `spsa_results.txt` for optimization results.

## 📄 License

MIT License
