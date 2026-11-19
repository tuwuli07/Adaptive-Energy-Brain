"""Display an overview of the Adaptive Energy Brain research project."""


def main():
    """Print the project description and current component status."""
    print("Adaptive Energy Brain")
    print("An AI-powered renewable energy management research project.")
    print("\nImplemented components:")

    implemented = (
        "Data preparation",
        "Energy Simulation",
        "Rule-based Baselines",
        "GNN-Transformer Forecasting (trained and evaluated research pipeline)",
    )
    for component in implemented:
        print(f"- {component}")

    print("\nCurrent phase:")
    print("- Deep Reinforcement Learning (next planned research phase; not implemented)")

    print("\nPlanned components:")
    planned = (
        "Deep Reinforcement Learning",
        "Meta-Learning Policy Selector",
        "DRO/CVaR Risk Optimization",
        "Dashboard",
        "Optional RAG Assistant",
    )
    for component in planned:
        print(f"- {component}")


if __name__ == "__main__":
    main()
