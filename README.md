# Cooperative Smart Traffic Management System Using Multi-Agent Reinforcement Learning


---

# Youtube Video

https://youtu.be/YoSbFgaoOPg

## Overview

Traffic congestion remains one of the most significant challenges in modern urban environments. Traditional traffic signal control systems often rely on fixed schedules that cannot adapt effectively to dynamic traffic conditions, resulting in increased delays, long queues, excessive fuel consumption, and higher emissions.

This project presents a Cooperative Smart Traffic Management System based on Multi-Agent Reinforcement Learning (MARL) for adaptive urban traffic signal control. The proposed framework combines Multi-Agent Proximal Policy Optimization (MAPPO), Graph Transformer communication, Gated Recurrent Unit (GRU) temporal modeling, and constrained reinforcement learning within a unified architecture.

The system was developed and evaluated using the Simulation of Urban Mobility (SUMO) environment on a realistic traffic network representing nine major signalized intersections in Assiut City, Egypt.

---

## Key Features

* Multi-Agent Proximal Policy Optimization (MAPPO)
* Centralized Training and Decentralized Execution (CTDE)
* Graph Transformer communication between neighboring intersections
* GRU-based temporal traffic modeling
* Multi-objective reward optimization
* Constrained Reinforcement Learning using Lagrangian optimization
* Realistic SUMO traffic simulation
* Scalable architecture for smart city deployment

---

## System Architecture

The proposed framework consists of the following modules:

1. Traffic Environment (SUMO)
2. State Representation Module
3. Graph Transformer Communication Layer
4. GRU Temporal Encoder
5. MAPPO Actor Network
6. Centralized Critic Network
7. Lagrangian Constraint Optimization Module

Traffic observations are collected from SUMO, processed through the Graph Transformer and GRU modules, and then used by MAPPO to determine optimal traffic signal actions.

---

## Technologies Used

* Python
* PyTorch
* SUMO
* TraCI
* NumPy
* Pandas
* Matplotlib

---

## Project Structure

```text
Project/
│
├── main.py
├── trainer.py
├── evaluator.py
├── env.py
├── reward.py
├── replay_buffer.py
├── config.py
│
├── models/
│   ├── actor.py
│   ├── critic.py
│   ├── graph_transformer.py
│   └── gru_encoder.py
│
├── sumo/
│   ├── assiut.net.xml
│   ├── assiut.rou.xml
│   └── assiut.sumocfg
│
├── results/
│
└── checkpoints/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install SUMO and ensure that the SUMO_HOME environment variable is configured correctly.

---

## Running the Project

Training:

```bash
python main.py
```

Evaluation:

```bash
python evaluator.py
```

---

## Performance Evaluation

The proposed MAPPO framework was compared against a traditional Fixed-Time Traffic Signal Control strategy.

### Results

| Metric           | MAPPO  | Fixed-Time |
| ---------------- | ------ | ---------- |
| Queue Length     | 1.50   | 9.10       |
| Delay (s)        | 1.84   | 6.68       |
| Waiting Time (s) | 18.92  | 164.50     |

### Improvement

| Metric       | Improvement |
| ------------ | ----------- |
| Queue Length | 83.5%       |
| Delay        | 72.5%       |
| Waiting Time | 88.5%       |

The results demonstrate significant reductions in congestion and waiting time while maintaining efficient traffic flow across the network.

---

## Research Contributions

* Development of a cooperative MAPPO-based traffic signal control framework.
* Integration of Graph Transformer communication for spatial coordination.
* Incorporation of GRU temporal encoding for historical traffic modeling.
* Multi-objective reward optimization considering congestion and throughput.
* Constrained reinforcement learning for sustainable traffic management.
* Evaluation on a realistic urban traffic network in Assiut City, Egypt.

---

## Future Work

* Real-time deployment using live traffic sensors.
* Integration with Vehicle-to-Infrastructure (V2I) communication.
* Large-scale city-wide traffic network evaluation.
* Multi-modal transportation optimization.
* Advanced Graph Neural Network architectures.

---

## License

This project was developed as part of Work-Based Professional Project course and IEEE IC-ESI 2026 Winning Project .

---

## Authors

Graduation Project Team : 

George Amgad -
Samaan Melad -
Amir Roshdy -
Mena Tharwot -
Maria Soliman -
Youstina Bassim -
Mahmoud Amr -
Amgad Ayman 

Faculty of Computers and Artificial Intelligence

Sphinx University

2026
