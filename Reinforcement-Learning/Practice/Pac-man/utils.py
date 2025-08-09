# Add More Helper Functions As Needed

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
