import detect
import fix
import process_models
import view
import webcam
from process_parser import get_args

args = get_args()


def main():
    if args.module == 'detect':
        detect.call()
        return

    if args.module == 'webcam':
        webcam.call()
        return

    if args.module == 'remake':
        process_models.remake()
        return

    if args.module == 'view':
        view.call(args.indices)
        return

    if args.module == 'fix':
        fix.call()
        return


if __name__ == '__main__':
    main()
