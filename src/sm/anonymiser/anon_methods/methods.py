import random
import string


def mask_value(field_value, field_type):
    """Creates a default masked value based on the given input type, if no type matches, return original value"""
    if field_type == bool:
        return False
    elif field_type == float:
        return 0.0
    elif field_type == int:
        return 0
    elif field_type == bytes:
        return b"0"
    elif field_type == str:
        return "****"
    else:
        return field_value


def perturb_value(field_value, field_type):
    """Takes a value and value type and adds random noise to the value, if no type matches, returns original value"""
    if field_type == bool:
        return bool(random.randint(0, 1))
    elif field_type == float:
        amount = 0.1  # 10%
        noise = (
            field_value * amount * (random.random() * 2 - 1)
        )  # (0 to 1)*2 -1   ->   -1 to 1
        return field_value + noise
    elif field_type == int:
        amount = 0.2  # 10%
        noise = (
            field_value * amount * (random.random() * 2 - 1)
        )  # (0 to 1)*2 -1   ->   -1 to 1
        return round(field_value + noise) + random.randint(0, 2) - 1
    elif field_type == bytes:
        new_field_value = bytearray(field_value)
        for x in range(len(new_field_value)):
            new_field_value[x] ^= random.randint(1, 255)  # XOR with random bitmask
        return bytes(new_field_value)
    elif field_type == str:
        new_field_value = ""
        field_val_len = len(field_value)
        length_change = random.randint(1, field_val_len * 2)
        real_letters = min(field_val_len, length_change)  # min:1, max:len
        additional_letters = max(0, length_change - field_val_len)
        for x in range(real_letters):
            new_field_value += chr(
                max(0, (ord(field_value[x]) + (random.randint(0, 10) - 5)))
            )
        for x in range(additional_letters):
            new_field_value += random.choice(string.ascii_letters + string.digits + "_")
        return new_field_value
    else:
        return field_value


def synth_value(synth, field_name, field_type):
    """Inputs a field name and type and uses the synthesiser to generate a value from the generator"""
    synth.generate_single_value(field_name, field_type)
