# __author__ = "Chang Gao"
# __copyright__ = "Copyright 2018 to the author"
# __license__ = "Private"
# __version__ = "0.1.0"
# __maintainer__ = "Chang Gao"
# __email__ = "chang.gao@uzh.ch"    `
# __status__ = "Prototype"
import importlib
import os
import errno


def main(args):
    # Create feature folder
    try:
        os.makedirs(os.path.join('feat', args.dataset_name))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Load modules according to dataset_name
    try:
        module_data = importlib.import_module('modules.' + args.dataset_name + '.feat_extractor')
        FeatExtractor = module_data.FeatExtractor
    except:
        raise RuntimeError('Please select a supported dataset.')

    # Extract Features
    extractor = FeatExtractor()
    extractor.extract(args)
