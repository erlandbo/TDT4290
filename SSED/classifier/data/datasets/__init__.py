

def dereference_dict(name:str):
    """
    Function to get dictionary to dereference
    output labels as numbers (from the model)
    to output labels as names.
    Need name of dataset of which to dereference
    """
    if name == "kauto5cls":
        kauto_dict = {
            0   :   "small",
            1   :   "medium",
            2   :   "large",
        }
        return kauto_dict
