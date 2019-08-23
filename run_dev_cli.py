from flask.cli import FlaskGroup
from pychunkedgraph.examples import create_example_app


app = create_example_app()
cli = FlaskGroup(create_app=create_example_app)


if __name__ == '__main__':
    cli()