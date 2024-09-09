import argparse

def true_with_float(value):
    if value.lower() == 'false':
        return False
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. If set to True, a float value is required.")
    


def valid_str_or_float(value):
    # Define a list of valid strings
    valid_strings = ['uniform', 'xavier']
    
    # Try to convert the value to a float
    try:
        return float(value)
    except ValueError:
        pass

    # Check if the value is in the list of valid strings
    if value.lower() in valid_strings:
        return value
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument: {value}. Must be a float or one of {valid_strings}.")



def valid_int_or_all(value):
    # Check if the value is the string "all"
    if value == "all":
        return value
    
    # Try to convert the value to an integer
    try:
        int_value = int(value)
        if int_value > 0:
            return int_value
        else:
            raise argparse.ArgumentTypeError(f"Invalid argument: {value}. Must be an integer greater than 0 or the string 'all'.")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid argument: {value}. Must be an integer greater than 0 or the string 'all'.")


def valid_int_list(value):
    try:
        # Split the string by commas and convert each part to an integer
        return [int(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid argument: {value}. Must be a comma-separated list of integers.")

