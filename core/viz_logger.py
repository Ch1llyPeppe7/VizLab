
"""
Visualization Data Logger — 结构化 JSON 日志

升级自原 scripts/base/viz_logger.py，更新路径配置和 metadata。

Usage:
    from core.viz_logger import VizLogger
    
    logger = VizLogger('pe_comparison', module='pe_analysis')
    logger.log_metric('kernel_value', 0.85, context={'delta': 100})
    logger.save()
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import warnings


PROJECT_ROOT = Path(__file__).parent.parent


class VizLogger:
    """结构化可视化数据记录器"""
    
    def __init__(self, viz_id: str, module: str = None, output_dir: Path = None):
        """
        Args:
            viz_id: 可视化唯一标识
            module: 所属模块（如 'pe_analysis', 'diff_geometry'）
            output_dir: 自定义输出目录
        """
        self.viz_id = viz_id
        self.module = module
        self.timestamp = datetime.now().isoformat()
        
        if output_dir is None:
            base = PROJECT_ROOT / "output" / "data_logs"
            output_dir = base / module if module else base
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            "script_version": "2.0",
            "numpy_version": np.__version__,
            "generated_by": "VizLab",
            "module": module
        }
        
    def set_description(self, description: str):
        self.summary["description"] = description
        
    def add_finding(self, finding: str, category: str = "general"):
        self.summary["key_findings"].append({
            "text": finding,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        
    def log_metric(self, name: str, value: Union[float, int, str], 
                   unit: str = None, context: Dict = None):
        metric_data = {"value": value, "unit": unit, "context": context or {}}
        if name in self.summary["metrics"]:
            if not isinstance(self.summary["metrics"][name], list):
                self.summary["metrics"][name] = [self.summary["metrics"][name]]
            self.summary["metrics"][name].append(metric_data)
        else:
            self.summary["metrics"][name] = metric_data
            
    def log_parameter(self, name: str, value: Any):
        self.data["parameters"][name] = self._serialize_value(value)
        
    def log_series(self, name: str, x: np.ndarray, y: np.ndarray,
                   x_label: str = None, y_label: str = None,
                   metadata: Dict = None):
        n = len(x)
        if n > 500:
            idx = np.linspace(0, n-1, 500, dtype=int)
            x, y = x[idx], y[idx]
        self.data["series"][name] = {
            "x": x.tolist(), "y": y.tolist(),
            "x_label": x_label, "y_label": y_label,
            "n_points": len(x),
            "statistics": {
                "x_range": [float(np.min(x)), float(np.max(x))],
                "y_range": [float(np.min(y)), float(np.max(y))],
                "y_mean": float(np.mean(y)),
                "y_std": float(np.std(y))
            },
            "metadata": metadata or {}
        }
        
    def log_matrix(self, name: str, matrix: np.ndarray, metadata: Dict = None):
        if matrix.shape[0] > 100 or matrix.shape[1] > 100:
            rs = max(1, matrix.shape[0] // 100)
            cs = max(1, matrix.shape[1] // 100)
            matrix = matrix[::rs, ::cs]
        self.data["matrices"][name] = {
            "shape": list(matrix.shape),
            "values": matrix.tolist(),
            "statistics": {
                "min": float(np.min(matrix)),
                "max": float(np.max(matrix)),
                "mean": float(np.mean(matrix)),
                "std": float(np.std(matrix)),
            },
            "metadata": metadata or {}
        }
        
    def log_array(self, name: str, array: np.ndarray, metadata: Dict = None):
        vals = array.tolist() if len(array) <= 1000 else array[::max(1, len(array)//1000)].tolist()
        self.data["arrays"][name] = {
            "shape": list(array.shape), "values": vals,
            "statistics": {
                "min": float(np.min(array)), "max": float(np.max(array)),
                "mean": float(np.mean(array)), "std": float(np.std(array)),
            },
            "metadata": metadata or {}
        }
        
    def log_figure(self, filepath: Union[str, Path], title: str = None,
                   fig_type: str = None):
        self.figures.append({
            "filename": Path(filepath).name,
            "path": str(filepath),
            "title": title,
            "type": fig_type,
            "created": datetime.now().isoformat()
        })
        
    def _serialize_value(self, value: Any) -> Any:
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
        print(f"  ✓ [VizLogger] Saved: {output_path}")
        return output_path


def quick_log(viz_id: str, description: str, module: str = None, **kwargs) -> VizLogger:
    """快速创建 logger"""
    logger = VizLogger(viz_id, module=module)
    logger.set_description(description)
    for k, v in kwargs.items():
        logger.log_parameter(k, v)
    return logger
