import sys
import file
import webcam
import process_models


def main():
    if len(sys.argv) != 2:
        print("Only one argument accepted (file, webcam, remake, f, w, r)")
        return

    if sys.argv[1] in ['f', 'file']:
        file.call()
        return

    if sys.argv[1] in ['w', 'webcam']:
        webcam.call()
        return

    if sys.argv[1] in ['r', 'remake']:
        process_models.remake()
        return


if __name__ == '__main__':
    main()
