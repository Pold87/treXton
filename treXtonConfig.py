import argparse

def parse_me():
    parser = argparse.ArgumentParser()

    parser.add_argument("-nt", "--num_textons", help="Size of texton dictionary", type=int, default=100)
    parser.add_argument("-mt", "--max_textons", help="Maximum amount of textons per image", type=int, default=500)
    parser.add_argument("-img", "--image", help="Path to image")
    parser.add_argument("-ts", "--texton_size", help="Size of the textons", type=int, default=5)
    
    args = parser.parse_args()

    return args
