"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="medical-terminology-helper", description="Medical students get a command-line assistant completing medical phrases and definitions, trained on public domain medical texts to reinforce terminology learning.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Medical Terminology Helper is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
