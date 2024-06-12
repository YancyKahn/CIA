import argparse

def calculate_ratio(expression):
    
    try:
        result = eval(expression)
        return result
    
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid expression: {e}")