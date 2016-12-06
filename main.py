# coding=utf-8
## Libraries to Import to run the project
from slack import main_slack


def main():
    print 'Begin'


if __name__ == '__main__':
    main()

    # Launch the agent (infinite loop)
    main_slack()