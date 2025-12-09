"""Entry point that runs the tool-calling demo."""

from demo import run_demo


def main() -> None:
    response = run_demo()
    print(response)


if __name__ == "__main__":
    main()
