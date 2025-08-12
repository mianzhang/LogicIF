"""
AST analyzer for code complexity and difficulty assessment.

This module provides functionality to analyze Python code structure and complexity
using Abstract Syntax Trees (AST) to determine difficulty levels.
"""

import ast
from typing import Dict, Any, Tuple


class ASTAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code structure and complexity."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Basic structure counts
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        
        # Control flow counts
        self.if_count = 0
        self.for_count = 0
        self.while_count = 0
        self.try_count = 0
        self.with_count = 0
        
        # Function call and operation counts
        self.call_count = 0
        self.return_count = 0
        self.assign_count = 0
        self.comparison_count = 0
        self.boolean_op_count = 0
        
        # Data structure usage
        self.list_count = 0
        self.dict_count = 0
        self.set_count = 0
        self.tuple_count = 0
        
        # Complexity indicators
        self.max_nesting_depth = 0
        self.current_depth = 0
        self.cyclomatic_complexity = 1  # Base complexity
        
        # Length analysis
        self.total_lines = 0
        self.non_empty_lines = 0
        self.function_lines = 0
        self.avg_function_length = 0
        
        # Collected information
        self.function_names = []
        self.imported_modules = []
        self.variable_names = set()
        self.builtin_functions_used = set()
        
        # Statistics tracking
        self.statistics_increments = []
        
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.function_names.append(node.name)
        
        # Analyze function parameters
        param_count = len(node.args.args)
        
        # Calculate function length (lines spanned by this function)
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            func_length = node.end_lineno - node.lineno + 1
            self.function_lines += func_length
        
        # Track nesting depth within function
        old_depth = self.current_depth
        self.current_depth = 0
        self.generic_visit(node)
        self.current_depth = old_depth
        
        return node
    
    def visit_ClassDef(self, node):
        self.class_count += 1
        self.generic_visit(node)
    
    def visit_Import(self, node):
        self.import_count += 1
        for alias in node.names:
            self.imported_modules.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        self.import_count += 1
        if node.module:
            self.imported_modules.append(node.module)
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.if_count += 1
        self.cyclomatic_complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()
    
    def visit_For(self, node):
        self.for_count += 1
        self.cyclomatic_complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()
    
    def visit_While(self, node):
        self.while_count += 1
        self.cyclomatic_complexity += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()
    
    def visit_Try(self, node):
        self.try_count += 1
        self.cyclomatic_complexity += len(node.handlers)  # Each except handler adds complexity
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()
    
    def visit_With(self, node):
        self.with_count += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()
    
    def visit_Call(self, node):
        self.call_count += 1
        
        # Track builtin function usage
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            builtin_names = {'len', 'max', 'min', 'sum', 'sorted', 'reversed', 'enumerate', 'zip', 'range', 'abs', 'round', 'int', 'float', 'str', 'list', 'dict', 'set', 'tuple'}
            if func_name in builtin_names:
                self.builtin_functions_used.add(func_name)
        
        # Check for statistics tracking patterns
        if isinstance(node.func, ast.Attribute) and node.func.attr in ['append', 'update']:
            # Look for dictionary updates that might be statistics
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if 'stat' in var_name.lower() or 'count' in var_name.lower():
                    self.statistics_increments.append({
                        'type': 'dict_update',
                        'variable': var_name,
                        'line': node.lineno
                    })
        
        self.generic_visit(node)
    
    def visit_Return(self, node):
        self.return_count += 1
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        self.assign_count += 1
        
        # Track variable names
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variable_names.add(target.id)
        
        # Check for statistics increment patterns
        if (isinstance(node.value, ast.BinOp) and 
            isinstance(node.value.op, ast.Add) and
            len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name)):
            
            var_name = node.targets[0].id
            if ('count' in var_name.lower() or 'stat' in var_name.lower() or
                var_name.startswith('cnt_') or var_name.startswith('num_')):
                self.statistics_increments.append({
                    'type': 'increment',
                    'variable': var_name,
                    'line': node.lineno
                })
        
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        self.comparison_count += 1
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        self.boolean_op_count += 1
        self.generic_visit(node)
    
    def visit_List(self, node):
        self.list_count += 1
        self.generic_visit(node)
    
    def visit_Dict(self, node):
        self.dict_count += 1
        self.generic_visit(node)
    
    def visit_Set(self, node):
        self.set_count += 1
        self.generic_visit(node)
    
    def visit_Tuple(self, node):
        self.tuple_count += 1
        self.generic_visit(node)
    
    def _enter_block(self):
        self.current_depth += 1
        self.max_nesting_depth = max(self.max_nesting_depth, self.current_depth)
    
    def _exit_block(self):
        self.current_depth -= 1
    
    def get_analysis_summary(self):
        """Return a comprehensive analysis summary."""
        # Calculate average function length
        self.avg_function_length = (self.function_lines / self.function_count) if self.function_count > 0 else 0
        
        return {
            # Basic structure
            'function_count': self.function_count,
            'class_count': self.class_count,
            'import_count': self.import_count,
            
            # Control flow
            'if_statements': self.if_count,
            'for_loops': self.for_count,
            'while_loops': self.while_count,
            'try_blocks': self.try_count,
            'with_statements': self.with_count,
            
            # Operations
            'function_calls': self.call_count,
            'return_statements': self.return_count,
            'assignments': self.assign_count,
            'comparisons': self.comparison_count,
            'boolean_operations': self.boolean_op_count,
            
            # Data structures
            'lists_created': self.list_count,
            'dicts_created': self.dict_count,
            'sets_created': self.set_count,
            'tuples_created': self.tuple_count,
            
            # Complexity metrics
            'max_nesting_depth': self.max_nesting_depth,
            'cyclomatic_complexity': self.cyclomatic_complexity,
            
            # Length metrics
            'total_function_lines': self.function_lines,
            'average_function_length': round(self.avg_function_length, 2),
            
            # Details
            'function_names': self.function_names,
            'imported_modules': list(set(self.imported_modules)),
            'unique_variables': len(self.variable_names),
            'builtin_functions_used': list(self.builtin_functions_used),
            'statistics_increments': self.statistics_increments,
            
            # Derived metrics
            'total_control_structures': self.if_count + self.for_count + self.while_count + self.try_count,
            'code_complexity_score': self._calculate_complexity_score()
        }
    
    def _calculate_complexity_score(self):
        """Calculate a simple complexity score based on various factors."""
        score = 0
        score += self.cyclomatic_complexity * 1  # Cyclomatic complexity has high weight
        score += self.max_nesting_depth * 3      # Deep nesting adds complexity
        score += self.call_count * 2           # Function calls add some complexity
        
        # Add function length factor - longer functions are generally more complex
        # Use average function length to normalize across different function counts
        avg_length = (self.function_lines / self.function_count) if self.function_count > 0 else 0
        score += avg_length * 0.5  # Length contributes to complexity but with lower weight
        
        return round(score, 2)


def analyze_function_ast(func_code: str) -> Dict[str, Any]:
    """
    Analyze a function's code using AST and return structural information.
    
    Args:
        func_code: String containing the function code
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(func_code)
        
        # Create analyzer and visit the tree
        analyzer = ASTAnalyzer()
        analyzer.visit(tree)
        
        # Get the analysis summary
        analysis = analyzer.get_analysis_summary()
        analysis['parsing_success'] = True
        analysis['parsing_error'] = None
        
        return analysis
        
    except SyntaxError as e:
        return {
            'parsing_success': False,
            'parsing_error': f"Syntax Error: {e}",
            'error_line': getattr(e, 'lineno', None),
            'error_text': getattr(e, 'text', None)
        }
    except Exception as e:
        return {
            'parsing_success': False,
            'parsing_error': f"Analysis Error: {e}",
            'error_type': type(e).__name__
        }


def determine_difficulty_from_ast(ast_analysis: Dict[str, Any]) -> Tuple[str, str]:
    """
    Determine difficulty level (easy, medium, hard) based on AST analysis metrics.
    Uses the code_complexity_score calculated by _calculate_complexity_score() method.
    
    Args:
        ast_analysis: Dictionary containing AST analysis results
        
    Returns:
        Tuple of (difficulty_level, reason)
    """
    if not ast_analysis.get('parsing_success', False):
        return "unknown", "AST parsing failed"
    
    # Primary metric: use the complexity score already calculated by _calculate_complexity_score()
    # This score combines: cyclomatic_complexity*1 + max_nesting_depth*3 + function_calls*2 + avg_length*0.5
    code_complexity_score = ast_analysis.get('code_complexity_score', 0)
    
    # Extract individual metrics for detailed reasoning and edge case handling
    cyclomatic_complexity = ast_analysis.get('cyclomatic_complexity', 1)
    max_nesting_depth = ast_analysis.get('max_nesting_depth', 0)
    function_calls = ast_analysis.get('function_calls', 0)
    average_function_length = ast_analysis.get('average_function_length', 0)
    total_control_structures = ast_analysis.get('total_control_structures', 0)
    
    # Primary difficulty determination based on complexity score thresholds
    # These thresholds are calibrated based on the _calculate_complexity_score() formula
    if code_complexity_score >= 30:
        difficulty = "hard"
        primary_reason = "high complexity score"
    elif code_complexity_score >= 15:
        difficulty = "medium" 
        primary_reason = "medium complexity score"
    elif code_complexity_score >= 8:
        difficulty = "easy"
        primary_reason = "low-medium complexity score"
    else:
        difficulty = "easy"
        primary_reason = "low complexity score"
    
    # Check for edge cases that might override the score-based classification
    edge_case_factors = []
    
    # Extremely deep nesting is always concerning
    if max_nesting_depth >= 5:
        edge_case_factors.append("very_deep_nesting")
        if difficulty != "hard":
            difficulty = "hard" if difficulty == "medium" else "medium"
    
    # Very high cyclomatic complexity
    if cyclomatic_complexity >= 15:
        edge_case_factors.append("very_high_cyclomatic_complexity")
        if difficulty != "hard":
            difficulty = "hard" if difficulty == "medium" else "medium"
    
    # Unusually long functions
    if average_function_length >= 80:
        edge_case_factors.append("very_long_function")
        if difficulty == "easy":
            difficulty = "medium"
    
    # Many control structures (complex logic flow)
    if total_control_structures >= 12:
        edge_case_factors.append("many_control_structures")
        if difficulty == "easy":
            difficulty = "medium"
    
    # Build detailed reason
    reason = f"{primary_reason}"
    if edge_case_factors:
        reason += f" + edge factors: {', '.join(edge_case_factors)}"
    
    # Add metric summary for transparency
    metrics_summary = f"(Score:{code_complexity_score:.1f}, CC:{cyclomatic_complexity}, Depth:{max_nesting_depth}, Calls:{function_calls}, Length:{average_function_length:.1f})"
    reason = f"{reason} {metrics_summary}"
    
    return difficulty, reason 