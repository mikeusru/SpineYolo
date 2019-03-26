import os
import ntpath
import argparse

parser = argparse.ArgumentParser(description='CreateGroundTruthFiles')
parser.add_argument('--validation',
                    action='store_true',
                    help='path to file with image paths and ground truth bounding boxes')

