"""
Visualization Data Logger - Structured JSON Output for LLM Analysis

This module provides logging capabilities for visualization scripts,
outputting structured JSON data alongside images for LLM analysis.

Usage:
    from base.viz_logger import VizLogger
    
    logger = VizLogger('my_visualization')
    logger.log_metric('kernel_value', 0.85, delta_q=100)
    logger.log_figure('heatmap.png', title='Kernel Matrix', shape=[50,50])
    logger.save()

Output Format:
    {
        "visualization_id": "my_visualization",
        "timestamp": "2026-02-25T22:30:00",
        "summary": {
            "description": "...",
            "key_findings": [...],
            "metrics": {...}
        },
        "data": {
            "parameters": {...},
            "series": {...},
            "matrices": {...}
        },
        "figures": [...],
        "metadata": {...}
    }
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import warnings


class VizLogger:
    """
    Logger for visualization data and metadata.
    
    Collects structured data during visualization generation
    and exports to JSON for LLM analysis.
    """
    
    def __init__(self, viz_id: str, output_dir: Path = None):
        """
        Initialize visualization logger.
        
        Args:
            viz_id: Unique identifier for this visualization
            output_dir: Directory for output JSON file (default: ../output/)
        """
        self.viz_id = viz_id
        self.timestamp = datetime.now().isoformat()
        
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "output" / "data_logs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.summary = {
            "description": "",
            "key_findings": [],
            "metrics": {}
        }
        
        self.data = {
            "parameters": {},
            "series": {},
            "matrices": {},
            "arrays": {}
        }
        
        self.figures = []
        self.metadata = {
            "script_version": "1.0",
            "numpy_version": np.__version__,
            "generated_by": "TCFMamba Visualization Suite"
        }
        
    def set_description(self, description: str):
        """Set visualization description."""
        self.summary["description"] = description
        
    def add_finding(self, finding: str, category: str = "general"):
        """
        Add a key finding.
        
        Args:
            finding: Description of the finding
            category: Category (e.g., 'theoretical', 'empirical', 'comparison')
        """
        self.summary["key_findings"].append({
            "text": finding,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_metric(self, name: str, value: Union[float, int, str], 
                   unit: str = None, context: Dict = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement (optional)
            context: Additional context (e.g., {'delta_q': 100, 'power': 3.0})
        """
        metric_data = {
            "value": value,
            "unit": unit,
            "context": context or {}
        }
        
        if name in self.summary["metrics"]:
            if not isinstance(self.summary["metrics"][name], list):
                self.summary["metrics"][name] = [self.summary["metrics"][name]]
            self.summary["metrics"][name].append(metric_data)
        else:
            self.summary["metrics"][name] = metric_data
            
    def log_parameter(self, name: str, value: Any):
        """Log a parameter used in the visualization."""
        self.data["parameters"][name] = self._serialize_value(value)
        
    def log_series(self, name: str, x: np.ndarray, y: np.ndarray,
                   x_label: str = None, y_label: str = None,
                   metadata: Dict = None):
        """
        Log a data series (e.g., for line plots).
        
        Args:
            name: Series name
            x: X values
            y: Y values
            x_label: Label for x-axis
            y_label: Label for y-axis
            metadata: Additional metadata
        """
        # Downsample if too large (keep at most 500 points)
        n_points = len(x)
        if n_points > 500:
            indices = np.linspace(0, n_points-1, 500, dtype=int)
            x = x[indices]
            y = y[indices]
            
        self.data["series"][name] = {
            "x": x.tolist(),
            "y": y.tolist(),
            "x_label": x_label,
            "y_label": y_label,
            "n_points": len(x),
            "statistics": {
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "y_mean": float(np.mean(y)),
                "y_std": float(np.std(y))
            },
            "metadata": metadata or {}
        }
        
    def log_matrix(self, name: str, matrix: np.ndarray,
                   row_labels: List[str] = None,
                   col_labels: List[str] = None,
                   metadata: Dict = None):
        """
        Log a matrix (e.g., for heatmaps).
        
        Args:
            name: Matrix name
            matrix: 2D numpy array
            row_labels: Labels for rows
            col_labels: Labels for columns
            metadata: Additional metadata
        """
        # Downsample large matrices (max 100x100)
        if matrix.shape[0] > 100 or matrix.shape[1] > 100:
            row_stride = max(1, matrix.shape[0] // 100)
            col_stride = max(1, matrix.shape[1] // 100)
            matrix = matrix[::row_stride, ::col_stride]
            
        self.data["matrices"][name] = {
            "shape": list(matrix.shape),
            "values": matrix.tolist(),
            "statistics": {
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix)),
                "std": float(np.std(matrix)),
                "diagonal_mean": float(np.mean(np.diag(matrix))) if matrix.shape[0] == matrix.shape[1] else None
            },
            "row_labels": row_labels,
            "col_labels": col_labels,
            "metadata": metadata or {}
        }
        
    def log_array(self, name: str, array: np.ndarray, metadata: Dict = None):
        """Log a 1D array with statistics."""
        self.data["arrays"][name] = {
            "shape": list(array.shape),
            "values": array.tolist() if len(array) <= 1000 else array[::max(1, len(array)//1000)].tolist(),
            "statistics": {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "median": float(np.median(array))
            },
            "metadata": metadata or {}
        }
        
    def log_figure(self, filepath: Union[str, Path], 
                   title: str = None,
                   fig_type: str = None,
                   shape: List[int] = None):
        """
        Log a generated figure.
        
        Args:
            filepath: Path to figure file
            title: Figure title
            fig_type: Type (e.g., 'heatmap', 'line_plot', 'animation')
            shape: Image dimensions [width, height]
        """
        filepath = Path(filepath)
        self.figures.append({
            "filename": filepath.name,
            "path": str(filepath),
            "title": title,
            "type": fig_type,
            "shape": shape,
            "created": datetime.now().isoformat()
        })
        
    def log_comparison(self, name: str, items: List[Dict]):
        """
        Log a comparison between multiple items.
        
        Args:
            name: Comparison name
            items: List of dicts with 'label', 'value', 'properties'
        """
        if "comparisons" not in self.data:
            self.data["comparisons"] = {}
        self.data["comparisons"][name] = items
        
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value to JSON-compatible format."""
        if isinstance(value, np.ndarray):
            return value.tolist() if value.size < 100 else f"<array shape={value.shape}>"
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        return value
        
    def save(self, filename: str = None) -> Path:
        """
        Save logged data to JSON file.
        
        Args:
            filename: Output filename (default: {viz_id}_data.json)
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            filename = f"{self.viz_id}_data.json"
            
        output_path = self.output_dir / filename
        
        output_data = {
            "visualization_id": self.viz_id,
            "timestamp": self.timestamp,
            "summary": self.summary,
            "data": self.data,
            "figures": self.figures,
            "metadata": self.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
        print(f"[VizLogger] Saved data log: {output_path}")
        return output_path
        
    def get_summary_text(self) -> str:
        """Get human-readable summary for LLM analysis."""
        lines = [
            f"=== {self.viz_id} ===",
            f"Description: {self.summary['description']}",
            "",
            "Key Findings:"
        ]
        
        for finding in self.summary["key_findings"]:
            lines.append(f"  - [{finding['category']}] {finding['text']}")
            
        if self.summary["metrics"]:
            lines.extend(["", "Key Metrics:"])
            for name, metric in self.summary["metrics"].items():
                if isinstance(metric, list):
                    values = [m['value'] for m in metric[:3]]
                    lines.append(f"  - {name}: {values}...")
                else:
                    unit_str = f" {metric['unit']}" if metric.get('unit') else ""
                    lines.append(f"  - {name}: {metric['value']}{unit_str}")
                    
        lines.extend(["", f"Figures generated: {len(self.figures)}", "Data series: " + ", ".join(self.data["series"].keys())])
        
        return "\n".join(lines)


class VizReportGenerator:
    """
    Generate comprehensive reports from multiple visualization logs.
    """
    
    def __init__(self, logs_dir: Path = None):
        if logs_dir is None:
            logs_dir = Path(__file__).parent.parent / "output" / "data_logs"
        self.logs_dir = Path(logs_dir)
        
    def collect_all_logs(self) -> List[Dict]:
        """Collect all visualization logs in directory."""
        logs = []
        if not self.logs_dir.exists():
            return logs
            
        for json_file in self.logs_dir.glob("*_data.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    logs.append(json.load(f))
            except Exception as e:
                warnings.warn(f"Failed to load {json_file}: {e}")
                
        return sorted(logs, key=lambda x: x.get("visualization_id", ""))
        
    def generate_master_report(self, output_path: Path = None) -> Path:
        """
        Generate master report combining all visualizations.
        
        Returns:
            Path to master report JSON
        """
        if output_path is None:
            output_path = self.logs_dir.parent / "visualization_master_report.json"
            
        logs = self.collect_all_logs()
        
        report = {
            "report_type": "TCFMamba Visualization Master Report",
            "generated_at": datetime.now().isoformat(),
            "total_visualizations": len(logs),
            "visualizations": [],
            "cross_analysis": {
                "themes": [],
                "comparisons": [],
                "recommendations": []
            }
        }
        
        # Extract key insights from each visualization
        for log in logs:
            viz_summary = {
                "id": log.get("visualization_id"),
                "description": log.get("summary", {}).get("description", ""),
                "key_findings": log.get("summary", {}).get("key_findings", []),
                "metrics": log.get("summary", {}).get("metrics", {}),
                "figures": [f.get("filename") for f in log.get("figures", [])],
                "data_series": list(log.get("data", {}).get("series", {}).keys()),
                "parameters": log.get("data", {}).get("parameters", {})
            }
            report["visualizations"].append(viz_summary)
            
        # Add cross-cutting analysis
        report["cross_analysis"]["themes"] = self._identify_themes(logs)
        report["cross_analysis"]["comparisons"] = self._extract_comparisons(logs)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"[VizReport] Generated master report: {output_path}")
        return output_path
        
    def _identify_themes(self, logs: List[Dict]) -> List[Dict]:
        """Identify common themes across visualizations."""
        themes = []
        
        # Check for frequency analysis theme
        freq_logs = [l for l in logs if "frequency" in l.get("visualization_id", "").lower()]
        if freq_logs:
            themes.append({
                "theme": "频率函数分析",
                "visualizations": [l.get("visualization_id") for l in freq_logs],
                "key_concept": "Power exponent 'a' determines frequency distribution and kernel decay",
                "related_formulas": ["ω_k = (-k/d)^a", "|K(Δq)| ~ O(1/(Δq)^(1/a))"]
            })
            
        # Check for kernel analysis theme
        kernel_logs = [l for l in logs if "kernel" in l.get("visualization_id", "").lower()]
        if kernel_logs:
            themes.append({
                "theme": "核函数特性",
                "visualizations": [l.get("visualization_id") for l in kernel_logs],
                "key_concept": "Positive definite kernels via Bochner theorem",
                "related_formulas": ["K(Δq) = Σ exp(i·ω_k·Δq)", "Re(K) = Σ cos(ω_k·Δq)"]
            })
            
        # Check for Kähler structure theme
        kaehler_logs = [l for l in logs if any(x in l.get("visualization_id", "").lower() 
                      for x in ["complex", "rotation", "kaehler"])]
        if kaehler_logs:
            themes.append({
                "theme": "凯勒结构",
                "visualizations": [l.get("visualization_id") for l in kaehler_logs],
                "key_concept": "Complex plane embedding z(q) = e^(i·ω·q) forms Kähler manifold",
                "related_formulas": ["z(q) = cos(ω·q) + i·sin(ω·q)", "⟨z₁, z₂⟩ = Σ z₁,k · z̄₂,k"]
            })
            
        return themes
        
    def _extract_comparisons(self, logs: List[Dict]) -> List[Dict]:
        """Extract comparison data across visualizations."""
        comparisons = []
        
        # Find all power exponent comparisons
        power_values = set()
        for log in logs:
            params = log.get("data", {}).get("parameters", {})
            if "power" in params or "a" in params:
                power = params.get("power", params.get("a"))
                if power is not None:
                    power_values.add(float(power))
                    
        if len(power_values) > 1:
            comparisons.append({
                "type": "幂指数对比",
                "values": sorted(list(power_values)),
                "observation": "不同幂指数a导致不同的频率分布和核函数衰减特性",
                "optimal_for_poi": "a=3 (TCFMamba) provides smooth decay suitable for POI recommendation"
            })
            
        return comparisons
        
    def generate_llm_prompt(self, output_path: Path = None) -> str:
        """
        Generate a prompt suitable for LLM analysis.
        
        Returns:
            Formatted prompt text
        """
        logs = self.collect_all_logs()
        
        prompt_lines = [
            "=== TCFMamba Visualization Analysis ===",
            "",
            "You are analyzing a series of mathematical visualizations from a POI recommendation research project.",
            "The visualizations demonstrate the theoretical foundations of LAPE (Location-Aware Position Encoding).",
            "",
            "## Core Mathematical Framework",
            "",
            "1. **Frequency Function**: ω_k = (-k/d)^a where a=3 for TCFMamba",
            "2. **Kernel Function**: K(Δq) = Σ exp(i·ω_k·Δq) via Bochner's theorem",
            "3. **Complex Embedding**: z(q) = e^(i·ω·q) forming Kähler structure",
            "4. **Asymptotic Decay**: |K(Δq)| ~ O(1/(Δq)^(1/a))",
            "",
            "## Visualizations Summary",
            ""
        ]
        
        for log in logs:
            viz_id = log.get("visualization_id", "unknown")
            summary = log.get("summary", {})
            
            prompt_lines.extend([
                f"### {viz_id}",
                f"Description: {summary.get('description', 'N/A')}",
                "Key Findings:"
            ])
            
            for finding in summary.get("key_findings", [])[:3]:  # Top 3 findings
                prompt_lines.append(f"  - [{finding.get('category', 'general')}] {finding.get('text', '')}")
                
            prompt_lines.append("")
            
        prompt_lines.extend([
            "## Analysis Tasks",
            "",
            "1. Summarize the theoretical contributions demonstrated by these visualizations",
            "2. Compare the impact of different power exponents (a=3 vs a=1 vs a=-1)",
            "3. Explain why TCFMamba's choice of a=3 is suitable for POI recommendation",
            "4. Identify the mathematical rigor in the progression from frequency function to kernel properties",
            "",
            "## Data Files",
            f"Total visualizations: {len(logs)}",
            f"Data logs location: {self.logs_dir}",
            ""
        ])
        
        prompt_text = "\n".join(prompt_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            print(f"[VizReport] Generated LLM prompt: {output_path}")
            
        return prompt_text


# Convenience function for quick logging
def quick_log(viz_id: str, description: str, **kwargs) -> VizLogger:
    """
    Quick logging helper.
    
    Usage:
        logger = quick_log("my_viz", "Description", power=3.0, dim=64)
        # ... generate visualization ...
        logger.save()
    """
    logger = VizLogger(viz_id)
    logger.set_description(description)
    
    for key, value in kwargs.items():
        logger.log_parameter(key, value)
        
    return logger
