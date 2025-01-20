import argparse


def validate_weight_init(value):
    parts = value.split()
    if len(parts) < 1:
        raise argparse.ArgumentTypeError(
            "Invalid format for --weight_init. Expected 'normal <mean> <std>', 'uniform', or 'fixed <mean> <std>'"
        )

    init_type = parts[0]

    # Default values
    default_mean = 0.0
    default_std = 0.02

    if init_type == "normal":
        if len(parts) == 1:
            return f"{init_type} {default_mean} {default_std}"
        elif len(parts) == 2:
            try:
                mean = float(parts[1])
                return f"{init_type} {mean} {default_std}"
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid mean value for 'normal'. Expected a float.")
        elif len(parts) == 3:
            try:
                mean = float(parts[1])
                std = float(parts[2])
                return f"{init_type} {mean} {std}"
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid mean or std value for 'normal'. Expected floats.")
        else:
            raise argparse.ArgumentTypeError("Invalid format for 'normal'. Expected: 'normal <mean> <std>'")

    elif init_type == "fixed":
        if len(parts) == 1:
            return f"{init_type} {default_mean} {default_std}"
        elif len(parts) == 2:
            try:
                mean = float(parts[1])
                return f"{init_type} {mean} {default_std}"
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid mean value for 'fixed'. Expected a float.")
        elif len(parts) == 3:
            try:
                mean = float(parts[1])
                std = float(parts[2])
                return f"{init_type} {mean} {std}"
            except ValueError:
                raise argparse.ArgumentTypeError("Invalid mean or std value for 'fixed'. Expected floats.")
        else:
            raise argparse.ArgumentTypeError("Invalid format for 'fixed'. Expected: 'fixed <mean> <std>'")

    elif init_type == "uniform":
        if len(parts) == 1:
            return f"{init_type} {default_mean} {default_std}"
        else:
            raise argparse.ArgumentTypeError("Invalid format for 'uniform'. Expected: 'uniform' with no additional parameters.")

    else:
        raise argparse.ArgumentTypeError("Invalid init_type. Expected 'normal', 'uniform', or 'fixed'.")

    return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def true_with_float(value):
    if value.lower() == 'false' or value is False:
        return False
    if value.lower() == 'true' or value is True:
        raise argparse.ArgumentTypeError("If set to True, a float value is required.")
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. If set to True, a float value is required.")



def valid_str_or_float(value):
    # Define a list of valid strings
    valid_strings = ['uniform', 'xavier', "randn"]
    
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
    if value is None:
        return None  # Explicitly handle None input
    try:
        # Split the string by commas, strip whitespace, and convert each part to an integer
        return [int(x.strip()) for x in value.split(',') if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid argument: {value}. Must be a comma-separated list of integers.")
