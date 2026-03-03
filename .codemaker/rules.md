# VizLab Project Rules

## Conda Environment Activation

**ALWAYS** activate the VizLab conda environment before running any Python command:

```bash
source /mnt/qianjin04-default/miniconda3/bin/activate && conda activate VizLab
```

This must be prepended to ALL terminal commands that involve Python execution in this project.

## Environment Details
- **Environment Name**: VizLab
- **Python Version**: 3.11
- **Key Dependencies**: numpy, scipy, matplotlib, seaborn, plotly, scikit-learn, umap-learn, kaleido, tqdm

## Working Directory
- All commands should be run from: `/mnt/qianjin04-default/Workspace/VizLab`

## Script Execution
- Use `python3 run.py <module>.<script>` to run individual analysis scripts
- Example: `python3 run.py pe_analysis.chaos`
