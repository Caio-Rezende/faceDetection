import argparse
import sys


samples_dir = "./assets/samples"


def get_args():
    parser = argparse.ArgumentParser(
        prog='faceDetection',
        description='Detects people', add_help="alert: the first detection will take longer for the models gonna be load to memory/gpu")

    subparsers = parser.add_subparsers(required=True, dest='module')

    subparsers.add_parser('print')

    parser_view = subparsers.add_parser('clear')
    parser_view.add_argument('threshold', type=int,
                             nargs='?',
                             default=1,
                             help='clear all models where matches are less then threshold')

    parser_view = subparsers.add_parser('view')
    parser_view.add_argument('indices', type=int,
                             nargs='+',
                             help='View the original file for the model index')
    parser_view.add_argument('--slow',
                             dest='slow',
                             action='store_true',
                             help='View the files 10 by 10')

    parser_train = subparsers.add_parser('train')
    add_processing_options(parser_train)
    parser_train.add_argument('--remake',
                              dest='remake',
                              action='store_true',
                              help='Resets the stored saved models')

    parser_webcam = subparsers.add_parser('webcam')
    add_processing_options(parser_webcam)
    add_save_options(parser_webcam)
    add_view_options(parser_webcam)

    parser_desktop = subparsers.add_parser('desktop')
    add_processing_options(parser_desktop)
    add_save_options(parser_desktop)
    add_view_options(parser_desktop)

    parser_detect = subparsers.add_parser('detect')
    add_processing_options(parser_detect)
    add_save_options(parser_detect)
    add_view_options(parser_detect)
    parser_detect.add_argument('path', type=str,
                               help='choose where the files are being read from', default=[samples_dir], nargs='*')

    parser_fix = subparsers.add_parser('fix')
    parser_fix.add_argument(
        '-i', '--indices', dest='indices', nargs='*', type=int, action='extend', default=[])
    parser_fix.add_argument(
        '-u', '--unknowns', dest='unknowns', nargs='*', type=int, action='extend', default=[])
    parser_fix.add_argument('-n', '--names', dest='names',
                            nargs='+', type=str, action='extend')

    del subparsers, parser_detect, parser_fix, parser_view, parser_train, parser_webcam

    return parser.parse_args(sys.argv[1:])


def add_processing_options(parser: argparse.ArgumentParser):
    parser.add_argument('--model', type=str,
                        dest='model',
                        choices=['cnn', 'hog'],
                        default="cnn",
                        help='Determine wich model the face_detection will use for it\'s algorithm')
    parser.add_argument('-t', '--tolerance',
                        dest='tolerance', type=float, default=0.35, help='The tolerance distance for the face detection to identify a match')
    parser.add_argument('-v', '--verbose',
                        dest='verbose', action='store_true', help='Display running info')


def add_save_options(parser: argparse.ArgumentParser):
    parser.add_argument('--save',
                        dest='save',
                        action='store_true',
                        help='Saves matches in models for next use')
    parser.add_argument('--save-unknown',
                        dest='unknown',
                        action='store_true',
                        help='Saves in models for next use even if it didn\'t match any')


def add_view_options(parser: argparse.ArgumentParser):
    parser.add_argument('--slow',
                        dest='slow',
                        action='store_true',
                        help='View the matches 10 by 10')
    parser.add_argument('--view',
                        dest='view',
                        action='store_true',
                        help='View the matches')
    parser.add_argument('-i', '--interactive',
                        dest='interactive',
                        action='store_true',
                               help='Allow to say who a face is when unkown - console interaction')
