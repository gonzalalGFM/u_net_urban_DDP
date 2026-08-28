# Distributed Training of U-Net Architecture for Data-Driven Aerodynamic Flow and Pollutant Dispersion Prediction in Urban Environments

Code repository for the research paper entitled "Distributed Training of U-Net Architecture for Data-Driven Aerodynamic Flow and Pollutant Dispersion Prediction in Urban Environments" with authors Alejandro González, Sergio Iserte, Maribel Castillo, and Sergio Chiva Vicent.

Example of both, python training scripts and bash scripts for launching the jobs, used in the manuscript can be found in the src/ folder, where the structure is as follow:

```
src/
├── bash_scripts/                  # Shell scripts to launch distributed jobs
│   ├── launch_ddp_amp.sh
│   ├── launch_ddp_compile.sh
│   ├── launch_ddp.sh
│   └── launch_ddp_workers_benchmark.sh
└── python_scripts/               # Training entrypoints
    ├── train_ddp_amp.py
    ├── train_ddp_compile.py
    ├── train_ddp.py
    └── train_ddp_workers_benchmark.py
```


# ABSTRACT
Computational fluid dynamics (CFD) is essential for analyzing urban aerodynamic flows and pollutant dispersion, yet its high computational cost limits practical deployment in operational and real-time applications. This paper presents a comprehensive methodology for developing a data-driven surrogate model to predict wind-driven odor dispersion in complex urban environments. We introduce an end-to-end pipeline that integrates physical domain digitalization, RANS-based CFD simulations, and systematic training-pipeline optimization on high-performance computing (HPC) infrastructure. Raw CFD outputs are interpolated onto structured 3D grids and decomposed into 2D horizontal slices, yielding a structured dataset used to design and optimize a U-Net surrogate architecture. The model is trained using distributed data parallelism via PyTorch, addressing scalability and computational efficiency challenges inherent in large-scale deep learning workflows. The target application focuses on forecasting odor plumes from a wastewater treatment plant within a densely built urban area in eastern Spain, covering a $\sim$2 $\times$ 1 km domain across 120 meteorological scenarios. Our results demonstrate that coupling advanced surrogate modeling with optimized distributed training pipelines can significantly accelerate CFD surrogate deployment while maintaining predictive fidelity. This work provides a scalable, reproducible framework for real-time urban microclimate and environmental impact assessment, bridging the gap between high-fidelity physics-based simulations and operational machine learning systems.

# REFERENCE
To be updated 
