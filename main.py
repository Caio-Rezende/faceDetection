#!/usr/bin/env python
import dlib

import actions.clear as clear
import actions.detect as detect
import actions.fix as fix
import actions.print as action_print
import actions.train as train
import actions.view as view
import actions.webcam as webcam

from process.args import get_args

args = get_args()


def main():
    if hasattr(args, 'model') and args.model == 'cnn':
        action_print.verbose("for cnn model, dlib could find devices: {}, cuda: {}".format(
            dlib.cuda.get_num_devices(), dlib.DLIB_USE_CUDA))

    if args.module == 'print':
        action_print.call()
        return

    if args.module == 'clear':
        clear.call()
        return

    if args.module == 'train':
        train.call()
        return

    if args.module == 'webcam':
        webcam.call()
        return

    if args.module == 'view':
        view.call(args.indices)
        return

    if args.module == 'fix':
        fix.call()
        return

    if args.module == 'detect':
        detect.call()
        return


if __name__ == '__main__':
    main()
