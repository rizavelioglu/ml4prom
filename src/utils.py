from termcolor import colored, cprint


def print_info(x: str) -> None:
    cprint(x, 'green')


def print_warning(x: str) -> None:
    cprint(x, 'red', attrs=['bold'])
