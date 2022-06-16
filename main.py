__author__ = "Chang Gao"
__copyright__ = "Copyright 2020"
__credits__ = ["Chang Gao", "Stefan Braun"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Chang Gao"
__email__ = "chang.gao@uzh.ch"
__status__ = "Prototype"

import os
import utils.argument as argument
from steps import data_prep, feat_ext, train, test
import importlib


class Project:
    def __init__(self, args):
        self.all_steps = ['data_prep', 'feat_ext', 'train', 'test']
        self.num_steps = len(self.all_steps)
        self.idx = self.all_steps.index(args.step)

    def step_in(self, args):
        if args.run_through:
            self.idx += 1
            args.step = self.all_steps[self.idx]


if __name__ == '__main__':
    # Process Arguments
    arg_parser = argument.ArgProcessor()
    args = arg_parser.get_args()

    # Use Default Feature File Paths if not specified
    try:
        module_log = importlib.import_module('modules.' + args.dataset_name + '.log')
    except:
        raise RuntimeError('Please select a supported dataset.')

    if args.trainfile is None or args.valfile is None or args.testfile is None:
        print("Loading from default feature file Paths...")
        _, args.trainfile, args.valfile, args.testfile = module_log.gen_dataset_name(args)

    args.trainfile = os.path.join('feat', args.dataset_name, args.trainfile)
    args.valfile = os.path.join('feat', args.dataset_name, args.valfile)
    args.testfile = os.path.join('feat', args.dataset_name, args.testfile)

    # Project Manager
    proj = Project(args)

    # Step 0 - Data Preparation
    if proj.idx == 0:
        print("####################################################################################################")
        print("# Step 0: Data Preparation                                                                         #")
        print("####################################################################################################")
        data_prep.main(args)
        proj.step_in(args)

    # Step 1 - Feature Extraction
    if proj.idx == 1:
        print("####################################################################################################")
        print("# Step 1: Feature Extraction                                                                       #")
        print("####################################################################################################")
        feat_ext.main(args=args)
        proj.step_in(args)

    # Step 2 - Train
    if proj.idx == 2:
        print("####################################################################################################")
        print("# Step 2: Train                                                                                 #")
        print("####################################################################################################")
        train.main(args)
        proj.step_in(args)

    # Step 3 - Test Pretrained Model
    if proj.idx == 3:
        print("####################################################################################################")
        print("# Step 3: Test                                                                            #")
        print("####################################################################################################")
        test.main(args)