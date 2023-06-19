import argparse
import sys


samples_dir = "./assets/samples"


def get_args():
    parser = argparse.ArgumentParser(
        prog='faceDetection',
        description='Detects people')

    subparsers = parser.add_subparsers(required=True, dest='module')

    subparsers.add_parser('webcam')
    subparsers.add_parser('remake')

    parser_view = subparsers.add_parser('view')
    parser_view.add_argument('indices', type=int,
                             nargs='+',
                             help='View the original file for the model index')
    parser_view.add_argument('--slow',
                             dest='slow',
                             action='store_true',
                             help='View the files 10 by 10')

    parser_detect = subparsers.add_parser('detect')
    parser_detect.add_argument('path', type=str,
                               help='choose where the files are being read from', default=[samples_dir], nargs='*')
    parser_detect.add_argument('--slow',
                               dest='slow',
                               action='store_true',
                               help='View the matches 10 by 10')
    parser_detect.add_argument('-v', '--view',
                               dest='view',
                               action='store_true',
                               help='View the matches')
    parser_detect.add_argument('--save',
                               dest='save',
                               action='store_true',
                               help='Saves matches in models for next use')
    parser_detect.add_argument('--save-unknown',
                               dest='unknown',
                               action='store_true',
                               help='Saves in models for next use even if it didn\'t match any')

    parser_fix = subparsers.add_parser('fix')
    parser_fix.add_argument(
        '-i', '--indices', dest='indices', nargs='*', type=int, action='extend')
    parser_fix.add_argument(
        '-u', '--unknowns', dest='unknowns', nargs='*', type=int, action='extend')
    parser_fix.add_argument('-n', '--names', dest='names',
                            nargs='+', type=str, action='extend')

    del parser_detect, parser_fix, parser_view

    return parser.parse_args(sys.argv[1:])
