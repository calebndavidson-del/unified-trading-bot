#!/usr/bin/env python3
"""
QuantConnect-Style Parameter Manager for Trading Bot
Provides automatic parameter range generation and combinations for optimization
"""

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime


@dataclass
class Parameter:
    """Individual parameter definition"""
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Union[int, float]
    current_value: Union[int, float] = None
    description: str = ""
    
    def __post_init__(self):
        if self.current_value is None:
            self.current_value = self.min_value
    
    @property
    def range_values(self) -> List[Union[int, float]]:
        """Generate all possible values for this parameter"""
        if isinstance(self.step, int) and isinstance(self.min_value, int):
            return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
        else:
            values = []
            current = self.min_value
            while current <= self.max_value:
                values.append(round(current, 4))  # Avoid floating point precision issues
                current += self.step
            return values
    
    def is_valid_value(self, value: Union[int, float]) -> bool:
        """Check if a value is within the parameter's valid range"""
        return self.min_value <= value <= self.max_value


class ParameterManager:
    """QuantConnect-style parameter manager for automated optimization"""
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.fixed_params: Dict[str, Any] = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'total_combinations': 0
        }
    
    def add_parameter(self, name: str, min_value: Union[int, float], 
                      max_value: Union[int, float], step: Union[int, float], 
                      description: str = "") -> 'ParameterManager':
        """
        QuantConnect-style parameter addition
        
        Args:
            name: Parameter name
            min_value: Minimum value for optimization
            max_value: Maximum value for optimization  
            step: Step size for value generation
            description: Human-readable description
            
        Returns:
            Self for method chaining
        """
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) cannot be greater than max_value ({max_value})")
        
        if step <= 0:
            raise ValueError(f"step ({step}) must be positive")
        
        self.parameters[name] = Parameter(
            name=name,
            min_value=min_value,
            max_value=max_value,
            step=step,
            description=description
        )
        
        # Update total combinations
        self._update_combinations_count()
        
        return self
    
    def add_fixed_parameter(self, name: str, value: Any) -> 'ParameterManager':
        """Add a fixed parameter that won't be optimized"""
        self.fixed_params[name] = value
        return self
    
    def get_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization"""
        if not self.parameters:
            return [self.fixed_params.copy()]
        
        # Get all parameter names and their possible values
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[name].range_values for name in param_names]
        
        # Generate all combinations
        combinations = []
        for value_combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, value_combination))
            # Add fixed parameters
            param_dict.update(self.fixed_params)
            combinations.append(param_dict)
        
        return combinations
    
    def get_total_combinations(self) -> int:
        """Get total number of parameter combinations"""
        return self.metadata['total_combinations']
    
    def set_current_values(self, values: Dict[str, Union[int, float]]) -> None:
        """Set current values for parameters"""
        for name, value in values.items():
            if name in self.parameters:
                if self.parameters[name].is_valid_value(value):
                    self.parameters[name].current_value = value
                else:
                    raise ValueError(f"Value {value} is not valid for parameter {name}")
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current parameter values including fixed parameters"""
        current = {name: param.current_value for name, param in self.parameters.items()}
        current.update(self.fixed_params)
        return current
    
    def create_smart_ranges(self, strategy_type: str = "rsi_bollinger") -> 'ParameterManager':
        """
        Create smart parameter ranges based on strategy type
        
        Args:
            strategy_type: Type of strategy to optimize
        """
        if strategy_type == "rsi_bollinger":
            # RSI Parameters
            self.add_parameter("rsi_period", 5, 30, 2, "RSI calculation period")
            self.add_parameter("rsi_oversold", 20, 40, 5, "RSI oversold threshold") 
            self.add_parameter("rsi_overbought", 60, 80, 5, "RSI overbought threshold")
            
            # Bollinger Band Parameters
            self.add_parameter("bb_period", 10, 40, 5, "Bollinger Bands period")
            self.add_parameter("bb_std", 1.5, 3.0, 0.25, "Bollinger Bands standard deviations")
            
            # Risk Management Parameters
            self.add_parameter("position_size", 0.05, 0.25, 0.05, "Position size as fraction of capital")
            self.add_parameter("stop_loss", 0.01, 0.05, 0.005, "Stop loss as fraction")
            
            # Fixed parameters that don't need optimization
            self.add_fixed_parameter("starting_capital", 100000)
            self.add_fixed_parameter("take_profit", 0.04)
            
        elif strategy_type == "momentum":
            # Momentum-specific parameters
            self.add_parameter("fast_ma", 5, 20, 2, "Fast moving average period")
            self.add_parameter("slow_ma", 20, 50, 5, "Slow moving average period")
            self.add_parameter("momentum_period", 10, 30, 5, "Momentum calculation period")
            self.add_parameter("position_size", 0.05, 0.20, 0.05, "Position size as fraction")
            
            # Fixed parameters
            self.add_fixed_parameter("starting_capital", 100000)
            self.add_fixed_parameter("stop_loss", 0.03)
            
        elif strategy_type == "mean_reversion":
            # Mean reversion parameters
            self.add_parameter("lookback_period", 10, 50, 5, "Mean reversion lookback period")
            self.add_parameter("entry_threshold", 1.5, 3.0, 0.25, "Entry threshold in standard deviations")
            self.add_parameter("exit_threshold", 0.5, 1.5, 0.25, "Exit threshold in standard deviations")
            self.add_parameter("position_size", 0.05, 0.15, 0.025, "Position size as fraction")
            
            # Fixed parameters
            self.add_fixed_parameter("starting_capital", 100000)
            self.add_fixed_parameter("stop_loss", 0.025)
        
        return self
    
    def _update_combinations_count(self):
        """Update the total combinations count"""
        if not self.parameters:
            self.metadata['total_combinations'] = 1
        else:
            count = 1
            for param in self.parameters.values():
                count *= len(param.range_values)
            self.metadata['total_combinations'] = count
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get comprehensive parameter information"""
        info = {
            'total_parameters': len(self.parameters),
            'fixed_parameters': len(self.fixed_params),
            'total_combinations': self.get_total_combinations(),
            'parameters': {},
            'fixed_params': self.fixed_params.copy(),
            'metadata': self.metadata.copy()
        }
        
        for name, param in self.parameters.items():
            info['parameters'][name] = {
                'min_value': param.min_value,
                'max_value': param.max_value,
                'step': param.step,
                'current_value': param.current_value,
                'description': param.description,
                'total_values': len(param.range_values),
                'sample_values': param.range_values[:5]  # First 5 values as sample
            }
        
        return info
    
    def save_to_file(self, filepath: str) -> None:
        """Save parameter configuration to file"""
        config = {
            'parameters': {
                name: {
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'step': param.step,
                    'current_value': param.current_value,
                    'description': param.description
                }
                for name, param in self.parameters.items()
            },
            'fixed_params': self.fixed_params,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load parameter configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Clear existing parameters
        self.parameters.clear()
        self.fixed_params.clear()
        
        # Load parameters
        for name, param_config in config.get('parameters', {}).items():
            self.add_parameter(
                name=name,
                min_value=param_config['min_value'],
                max_value=param_config['max_value'],
                step=param_config['step'],
                description=param_config.get('description', '')
            )
            if 'current_value' in param_config:
                self.parameters[name].current_value = param_config['current_value']
        
        # Load fixed parameters
        self.fixed_params.update(config.get('fixed_params', {}))
        
        # Load metadata
        self.metadata.update(config.get('metadata', {}))
    
    def validate_combination(self, combination: Dict[str, Any]) -> bool:
        """Validate that a parameter combination is valid"""
        for name, value in combination.items():
            if name in self.parameters:
                if not self.parameters[name].is_valid_value(value):
                    return False
        return True
    
    def get_optimization_summary(self) -> str:
        """Get a human-readable summary of the optimization setup"""
        total_params = len(self.parameters)
        total_combinations = self.get_total_combinations()
        
        summary = f"Parameter Optimization Setup:\n"
        summary += f"- {total_params} parameters to optimize\n"
        summary += f"- {len(self.fixed_params)} fixed parameters\n"
        summary += f"- {total_combinations:,} total combinations to test\n\n"
        
        summary += "Parameters to optimize:\n"
        for name, param in self.parameters.items():
            values_count = len(param.range_values)
            summary += f"  • {name}: {param.min_value} to {param.max_value} (step {param.step}) = {values_count} values\n"
        
        if self.fixed_params:
            summary += "\nFixed parameters:\n"
            for name, value in self.fixed_params.items():
                summary += f"  • {name}: {value}\n"
        
        return summary


# Convenience function for quick setup
def create_default_parameters(strategy_type: str = "rsi_bollinger") -> ParameterManager:
    """Create a ParameterManager with smart default ranges"""
    manager = ParameterManager()
    return manager.create_smart_ranges(strategy_type)


# Example usage
if __name__ == "__main__":
    # Example 1: Manual parameter setup
    params = ParameterManager()
    params.add_parameter("rsi_period", 10, 25, 2)
    params.add_parameter("bb_period", 15, 35, 5)
    params.add_parameter("position_size", 0.05, 0.25, 0.05)
    params.add_fixed_parameter("starting_capital", 100000)
    
    print("Manual Setup:")
    print(params.get_optimization_summary())
    print(f"Total combinations: {params.get_total_combinations()}")
    
    # Example 2: Smart ranges
    smart_params = create_default_parameters("rsi_bollinger")
    print("\nSmart RSI/Bollinger Setup:")
    print(smart_params.get_optimization_summary())